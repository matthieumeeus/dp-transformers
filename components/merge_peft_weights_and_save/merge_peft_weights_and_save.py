# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

'''Load and merge peft weights to a base model and save the merged model.'''

import sys
import logging
import transformers
from peft import PeftModel

from dataclasses import dataclass, field
from typing import Union
from pathlib import Path


logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: Union[str, Path] = field(default="gpt2", metadata={
        "help": "Model name in HuggingFace, e.g. 'gpt2'"
    })
    peft_weights_path: Union[str, Path] = field(default=".", metadata={
        "help": "Path to peft weights to merge with base model and save"
    })
    output_dir: str = field(default=".", metadata={
        "help": "Path to output directory"
    })


@dataclass
class Arguments:
    model: ModelArguments


def main(args: Arguments):
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = logging.INFO
    logger.setLevel(log_level)

    logger.info(f"Model parameters {args.model}")

    # Load the base model
    logger.info(f"Loading the base model from: {args.model.model_name_or_path}")
    base_model = transformers.AutoModelForCausalLM.from_pretrained(str(args.model.model_name_or_path))
    
    # Load the peft model
    logger.info(f"Loading the peft weights from: {args.model.peft_weights_path}")
    peft_model = PeftModel.from_pretrained(base_model, str(args.model.peft_weights_path))

    # Merge the peft weights to the base model and save the merged model
    merged_model = peft_model.merge_and_unload()
    merged_model.save_pretrained(args.model.output_dir)

    # Load and save the tokenizer
    logger.info(f"Loading the tokenizer from: {args.model.model_name_or_path}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(args.model.model_name_or_path))
    tokenizer.save_pretrained(args.model.output_dir)


if __name__ == "__main__":
    arg_parser = transformers.HfArgumentParser((ModelArguments))
    model_args, = arg_parser.parse_args_into_dataclasses()
    main(Arguments(model=model_args))
