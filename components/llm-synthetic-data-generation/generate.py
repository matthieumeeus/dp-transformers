# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

'''Generate synthetic data samples from an LLM fine-tuned with QLoRA.'''

import csv
import datasets
import transformers
import sys
import logging
import torch
from accelerate import Accelerator

from pynvml import *

from dataclasses import dataclass, field
from typing import Optional, Union
from pathlib import Path

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
    batch_size: int = field(default=32, metadata={
        "help": "Batch size"
    })
    seed: int = field(default=42, metadata={
        "help": "Random seed"
    })
    enable_lora: bool = field(default=False, metadata={
        "help": "Whether to enable LoRA"
    })
    lora_path: Union[str, Path] = field(default=".", metadata={
        "help": "Path to loaded lora weight if any"
    })
    mixed_precision: str = field(default="no", metadata={
        "help": "Kind of mixed precision used during training (fp16, bf16, no)"
    })
    load_dtype: str = field(default="fp32", metadata={
        "help": "Non-quantized model parameters load dtype (fp32, fp16, bf16)"
    })
    max_new_tokens: int = field(default=128, metadata={
        "help": "Maximum number of tokens to generate"
    })
    output_dir: str = field(default=".", metadata={
        "help": "Path to output directory"
    })


@dataclass
class DataArguments:
    train_data_path: Optional[Path] = field(default=None, metadata={
        "help": "Path to training data in jsonl format"
    })


@dataclass
class Arguments:
    model: ModelArguments
    data: DataArguments


def main(args: Arguments):
    transformers.set_seed(args.model.seed)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = logging.INFO
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info(f"Model parameters {args.model}")

    accelerator = Accelerator(mixed_precision=args.model.mixed_precision)

    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model.model_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    # Load dataset
    dataset = datasets.DatasetDict({
            "train": datasets.Dataset.from_json(str(args.data.train_data_path)),
        })

    # Tokenize data
    def preprocess_function(examples):
        model_inputs = tokenizer(examples['prompt'], padding=False)
        return model_inputs

    # Tokenize data
    with accelerator.main_process_first():
        dataset = dataset.map(
            preprocess_function, batched=True, num_proc=None, desc="tokenizing dataset", 
            remove_columns=dataset.column_names['train']
        )

    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load model
    logger.info(f"Loading model: {args.model.model_name_or_path}")
    if args.model.load_dtype == "fp32":
        torch_dtype = torch.float32
    elif args.model.load_dtype == "fp16":
        torch_dtype = torch.float16
    elif args.model.load_dtype == "bf16":
        torch_dtype = torch.bfloat16
    else:
        raise ValueError(f"Invalid load dtype {args.model.load_dtype}")
    model = transformers.AutoModelForCausalLM.from_pretrained(str(args.model.model_name_or_path), quantization_config=bnb_config, torch_dtype=torch_dtype)
    
    if args.model.enable_lora:
        logger.info("Loading LoRA")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.model.lora_path)
    else:
        logger.info("Not loading LoRA")

    model.eval()
    model = accelerator.prepare(model)
    generation_kwargs = {"max_new_tokens": args.model.max_new_tokens, "pad_token_id": tokenizer.pad_token_id, 
                             "eos_token_id": tokenizer.eos_token_id, "num_return_sequences": 1,
                             "do_sample": True, "top_p": 0.95, "temperature": 1.0}
    dataset.set_format(type="torch")

    with accelerator.split_between_processes(dataset["train"]["input_ids"]) as prompt:
        all_prompts = []
        all_generations = []

        # in case we have fewer examples than bs
        batch_size = min(len(prompt), args.model.batch_size)

        for i in range(0, len(prompt), batch_size):
            # prevent overflow if query tensors are not even multiple of bs
            end_index = min(len(prompt), i + batch_size)

            batch = prompt[i:end_index]
            batch_mask = [torch.ones_like(element) for element in batch]
            inputs = {"input_ids": batch, "attention_mask": batch_mask}

            padded_inputs = tokenizer.pad(
                inputs,
                padding=True,
                max_length=None,
                pad_to_multiple_of=None,
                return_tensors="pt",
            ).to(accelerator.device)

            with torch.no_grad():
                generations = model.generate(**padded_inputs, **generation_kwargs)

            for generation, mask in zip(generations, padded_inputs["attention_mask"]):
                output = generation[(1 - mask).sum() :]  # remove padding
                p = output[:(mask).sum()] # get prompt
                g = output[(mask).sum():]  # remove prompt
                all_prompts.append(p)
                all_generations.append(g)
        
        all_prompts_decoded = [tokenizer.decode(r.squeeze(), skip_special_tokens=True) 
                                    for r in all_prompts]
        all_generations_decoded = [tokenizer.decode(r.squeeze(), skip_special_tokens=True) 
                                    for r in all_generations]
        
        output_path = os.path.join(args.model.output_dir, f"generations_gpu{accelerator.device.index}.csv")
        with open(output_path, 'w', newline='', encoding="utf-8") as wf:
            csv_writer = csv.writer(wf)
            csv_writer.writerow(["Prompt", "Generation"])
            for obj in zip(all_prompts_decoded, all_generations_decoded):
                csv_writer.writerow([obj[0], obj[1]])
                    
    print_gpu_utilization()


if __name__ == "__main__":
    arg_parser = transformers.HfArgumentParser((ModelArguments, DataArguments))
    model_args, data_args = arg_parser.parse_args_into_dataclasses()
    main(Arguments(model=model_args, data=data_args))
