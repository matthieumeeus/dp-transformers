# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

'''Select a checkpoint and copy it to an output directory.'''

import sys
import logging
import shutil
import transformers

from dataclasses import dataclass, field
from typing import Union
from pathlib import Path


logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    input_dir: Union[str, Path] = field(default=".", metadata={
        "help": "Path to input directory"
    })
    checkpoint_number: int = field(default="1", metadata={
        "help": "Which checkpoint to use (e.g. saved after which epoch)"
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

    # List the folders in the input_dir
    input_dir = Path(args.model.input_dir)
    checkpoint_folders = [f for f in input_dir.iterdir() if f.is_dir()]

    # Sort the folders (looks like checkpoint-10545)
    checkpoint_folders.sort(key=lambda x: int(x.name.split("-")[1]))
    logger.info(f"Checkpoint folders: {checkpoint_folders}")

    # Select the checkpoint
    checkpoint_folder = checkpoint_folders[args.model.checkpoint_number - 1]

    # Get each file in the checkpoint_folder
    checkpoint_files = list(checkpoint_folder.glob("*"))

    # Copy checkpoint_files to output_dir
    for file in checkpoint_files:
        shutil.copy(file, args.model.output_dir)


if __name__ == "__main__":
    arg_parser = transformers.HfArgumentParser((ModelArguments))
    model_args, = arg_parser.parse_args_into_dataclasses()
    main(Arguments(model=model_args))
