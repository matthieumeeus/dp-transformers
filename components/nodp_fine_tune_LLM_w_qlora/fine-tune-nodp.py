# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

'''Train LLMs without DP using QLoRA'''

import datasets
import dp_transformers
import transformers
import sys
import logging
import torch
import ast
from data_utils import MyDataset, MyChatDataset

from pynvml import *

from dataclasses import dataclass, field, asdict
from typing import Optional, Union, List
from pathlib import Path

from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training

def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: Union[str, Path] = field(default="gpt2", metadata={
        "help": "Model name in HuggingFace, e.g. 'gpt2'"
    })
    sequence_len: int = field(default=128, metadata={
        "help": "Maximum sequence length"
    })


@dataclass
class DataArguments:
    train_data_path: Optional[Path] = field(default=None, metadata={
        "help": "Path to training data in jsonl format"
    })
    eval_data_path: Optional[Path] = field(default=None, metadata={
        "help": "Path to evaluation data in jsonl format"
    })
    chat_format: bool = field(default=False, metadata={
        "help": "Whether the dataset should be processed chat format or not"
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

def main(args: Arguments):
    transformers.set_seed(args.train.seed)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = args.train.get_process_log_level()
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
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model.model_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load dataset
    if args.data.chat_format:
        logger.info(f"Loading a dataset in a chat format (e.g. system-user-assistant)")
        dataset = MyChatDataset(args.data.train_data_path, tokenizer, args.model.sequence_len)
    else:
        logger.info(f"Loading a dataset in a plain format (i.e. prompt-completion)")
        dataset = MyDataset(args.data.train_data_path, tokenizer, args.model.sequence_len)

    # Tokenize data
    with train_args.main_process_first(desc="tokenizing dataset"):
        dataset.dataset = dataset.dataset.map(
            dataset.preprocess_function, batched=True, num_proc=8, desc="tokenizing dataset", 
            remove_columns=dataset.dataset.column_names['train']
        )

    # do the same for the eval dataset if provided
    if args.data.eval_data_path is not None:
        if args.data.chat_format:
            eval_dataset = MyChatDataset(args.data.eval_data_path, tokenizer, args.model.sequence_len)
        else:
            eval_dataset = MyDataset(args.data.eval_data_path, tokenizer, args.model.sequence_len)
        with train_args.main_process_first(desc="tokenizing dataset"):
            eval_dataset.dataset = eval_dataset.dataset.map(
                eval_dataset.preprocess_function, batched=True, num_proc=8, desc="tokenizing dataset", 
                remove_columns=eval_dataset.dataset.column_names['train']
            )
    
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load model
    logger.info(f"Loading model: {args.model.model_name_or_path}")
    model = transformers.AutoModelForCausalLM.from_pretrained(str(args.model.model_name_or_path), quantization_config=bnb_config)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=train_args.gradient_checkpointing)

    if args.lora.enable_lora:
        logger.info("Using LoRA")
        model = get_peft_model(model=model, peft_config=args.lora.as_peft_config())
    else:
        logger.info("Not using LoRA")

    if args.train.local_rank == 0:
        logger.info(f"Total number of parameters of the model: {model.num_parameters(only_trainable=False)}")
        logger.info(f"Fine-tuned number of parameters of the model: {model.num_parameters(only_trainable=True)}")

    trainer = transformers.Trainer(
        args=args.train,
        model=model,
        train_dataset=dataset.dataset['train'],
        eval_dataset=eval_dataset.dataset['train'] if args.data.eval_data_path is not None else None,
        tokenizer=tokenizer
    )

    result = trainer.train()
    final_evaluation_results = trainer.evaluate()

    def print_summary(result):
        print(f"Time: {result.metrics['train_runtime']:.2f}")
        print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
        print("Final evaluation results:", final_evaluation_results)
        print_gpu_utilization()

    print_summary(result)

    trainer.save_model()


if __name__ == "__main__":
    arg_parser = transformers.HfArgumentParser(
        (dp_transformers.TrainingArguments, ModelArguments, LoraArguments, DataArguments)
    )
    train_args, model_args, lora_args, data_args = arg_parser.parse_args_into_dataclasses()
    main(Arguments(train=train_args, model=model_args, lora=lora_args, data=data_args))
