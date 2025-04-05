# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

'''Train LLMs without DP using QLoRA'''
import os
import datasets
import dp_transformers
import transformers
import sys
import logging
import torch
import ast

from pynvml import *

from dataclasses import dataclass, field, asdict
from typing import Optional, Union, List
from pathlib import Path

from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from utils import internal_prepare_model_for_kbit_training


def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


logger = logging.getLogger(__name__)


TORCH_DTYPES = {
    "fp16": torch.float16,
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
}


@dataclass
class ModelArguments:
    model_name_or_path: Union[str, Path] = field(default="gpt2", metadata={
        "help": "Model name in HuggingFace, e.g. 'gpt2'"
    })
    quantization_4bit: bool = field(default=True, metadata={
        "help": "Whether to apply 4bit quantization for the base model."
    })
    torch_dtype: Optional[str] = field(default=None, metadata={
        "help": "The torch dtype to use for the model (None: the default dtype of the model; fp16; bf16.)"
    })
    trust_remote_code: bool = field(default=False, metadata={
        "help": "Whether to trust remote code when loading model from HuggingFace."
    })
    use_flash_attention_2: bool = field(default=True, metadata={
        "help": "Whether to use flash_attention_2 for the model."
    })

    def __post_init__(self):
        if self.torch_dtype is not None and isinstance(self.torch_dtype, str):
            if self.torch_dtype not in TORCH_DTYPES:
                 raise ValueError(f"Invalid torch dtype: {self.torch_dtype}. Must be one of {list(TORCH_DTYPES.keys())}")
            self.torch_dtype = TORCH_DTYPES[self.torch_dtype]

 
@dataclass
class DataArguments:
    tokenized_train_data_path: Optional[Path] = field(default=None, metadata={
        "help": "Path to tokenized data in HF dataset format"
    })
    num_samples: int = field(default=0, metadata={
        "help": "Number of samples to use from the dataset. 0 means using all samples."
    })
    tokenized_validation_data_path: Optional[Path] = field(default=None, metadata={
        "help": "Path to tokenized validation data in HF dataset format"
    })


@dataclass
class LoraArguments:
    enable_lora: bool = field(default=False, metadata={
        "help": "Whether to enable LoRA"
    })
    lora_dim: int = field(default=8, metadata={
        "help": "LoRA dimension"
    })
    lora_alpha: int = field(default=8, metadata={
        "help": "LoRA alpha"
    })
    lora_dropout: float = field(default=0.0, metadata={
        "help": "LoRA dropout"
    })

    target_modules: List[str] = field(
        default_factory=list,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with Lora."
            "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )

    def as_peft_config(self) -> LoraConfig:
        if not self.enable_lora:
            raise ValueError("LoRA is not enabled, cannot convert to LoRA config")
        params = asdict(self)
        params.pop("enable_lora")
        params["r"] = params.pop("lora_dim")
        params["target_modules"] = ast.literal_eval(params["target_modules"][0])
        return LoraConfig(**params)


@dataclass
class Arguments:
    train: dp_transformers.TrainingArguments
    model: ModelArguments
    lora: LoraArguments
    data: DataArguments
    privacy: dp_transformers.PrivacyArguments


def load_model(model_args: ModelArguments, lora_args: LoraArguments, gradient_checkpointing: bool,
               ) -> transformers.PreTrainedModel:
    logger.info(f"Loading model: {model_args.model_name_or_path}")
    model_kwargs = dict()

    if model_args.use_flash_attention_2:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    if model_args.quantization_4bit:
        # For bnb_4bit_compute_dtype check if GPU supports bf16 and if not use fp16
        bnb_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )
        model_kwargs["quantization_config"] = bnb_config

    model = transformers.AutoModelForCausalLM.from_pretrained(str(model_args.model_name_or_path),
                                                                torch_dtype=model_args.torch_dtype,
                                                                trust_remote_code=model_args.trust_remote_code,
                                                                **model_kwargs)


    if model_args.quantization_4bit:
        # Normally you'd need to call prepare_model_for_kbit_training from peft
        # model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.train.gradient_checkpointing)
        # But it casts all non INT8 parameters to fp32 and that leads to huge memory consumption so we remove that part
        model = internal_prepare_model_for_kbit_training(model, use_gradient_checkpointing=gradient_checkpointing)
    else:
        model = model.cuda()

    if lora_args.enable_lora:
        logger.info("Using LoRA")
        if not model_args.quantization_4bit and gradient_checkpointing:
            model.enable_input_require_grads()
        model = get_peft_model(model=model, peft_config=lora_args.as_peft_config())
    else:
        logger.info("Not using LoRA")

    return model


def load_tokenizer(model_args: ModelArguments) -> transformers.PreTrainedTokenizer:
    logger.info(f"Loading tokenizer: {model_args.model_name_or_path}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return tokenizer

def main(args: Arguments):
    transformers.set_seed(args.train.seed)

    distributed_state = args.train.distributed_state

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = args.train.get_process_log_level()
    logging.getLogger().setLevel(level=log_level)
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {args.train.local_rank}, device: {args.train.device}, n_gpu: {args.train.n_gpu}, "
        f"distributed training: {bool(args.train.local_rank != -1)}, 16-bits training: {args.train.fp16}"
    )
    logger.info(f"Training/evaluation parameters {args.train}")
    logger.info(f"Model parameters {args.model}")

    # Load tokenizer
    tokenizer = load_tokenizer(args.model)
    
    # Save model id
    training_metadata = dp_transformers.TrainingMetadata.from_pretrained(args.model.model_name_or_path)
    training_metadata.model_history.append("fine_tuning")

    # Load dataset
    train_data = datasets.load_from_disk(args.data.tokenized_train_data_path)
    val_data = datasets.load_from_disk(args.data.tokenized_validation_data_path) if args.data.tokenized_validation_data_path else None

    if args.data.num_samples > 0:
        # Shuffle and take the first num_samples
        train_data = train_data.shuffle(seed=args.train.seed, keep_in_memory=True).select(range(args.data.num_samples), keep_in_memory=True)

    # Load model
    logger.info(f"Loading model: {args.model.model_name_or_path}")
    model = load_model(model_args=args.model, lora_args=args.lora, gradient_checkpointing=args.train.gradient_checkpointing)

    if distributed_state.is_main_process:
        logger.info(f"Total number of parameters of the model: {model.num_parameters(only_trainable=False)}")
        logger.info(f"Fine-tuned number of parameters of the model: {model.num_parameters(only_trainable=True)}")
        logger.debug(f"Environment variables: {os.environ}")

    if args.train.gradient_checkpointing:
        logger.info("Set ddp_find_unused_parameters to False for gradient checkpointing")
        args.train.ddp_find_unused_parameters = False

    model = model.cuda()

    trainer = dp_transformers.dp_utils.OpacusDPTrainer(
        args=args.train,
        model=model,
        privacy_args=args.privacy,
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=tokenizer,
    )

    if len(args.train.fsdp):
        assert (len(args.train.fsdp) > 0) == (trainer.accelerator.state.fsdp_plugin is not None)
        assert (len(args.train.fsdp) > 0) == trainer.is_fsdp_enabled
        logger.info(f"FSDP Plugin: {trainer.accelerator.state.fsdp_plugin}")

    # flush
    sys.stdout.flush()
    sys.stderr.flush()
    distributed_state.wait_for_everyone()

    result = trainer.train()

    def print_summary(result):
        print(f"Time: {result.metrics['train_runtime']:.2f}")
        print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
        print_gpu_utilization()

    print_summary(result)

    if distributed_state.is_main_process:
        logger.warning(f"Final epsilon: {trainer.get_prv_epsilon()}")
        logger.info("Saving model")
        trainer.save_model()
        training_metadata.save_pretrained(args.train.output_dir)

    distributed_state.wait_for_everyone()
    logger.info("Training completed. Exiting...")


if __name__ == "__main__":
    arg_parser = transformers.HfArgumentParser(
        (dp_transformers.TrainingArguments, ModelArguments, LoraArguments, DataArguments, dp_transformers.PrivacyArguments)
    )
    train_args, model_args, lora_args, data_args, privacy_args = arg_parser.parse_args_into_dataclasses()
    main(Arguments(train=train_args, model=model_args, lora=lora_args, data=data_args, privacy=privacy_args))
