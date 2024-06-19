
'''Trained roberta sentiment classifier on the synthetic data to compute utility'''

from pathlib import Path
import numpy as np
import transformers
import datasets
import sys
import logging
from mia_methods import compute_mia_score

from dataclasses import dataclass, field
from typing import Optional, Union

logger = logging.getLogger(__name__)

@dataclass
class DataArguments:
    inference_path: Path = field(default=None, metadata={
        "help": "Path to inference data"
    })
    synthetic_path: Path = field(default=None, metadata={
        "help": "Path to synthetic data"
    })
    text_name: str = field(default="sentence", metadata={
        "help": "Name of the text column in the dataset"
    })
    label_name: str = field(default="label", metadata={
        "help": "Name of the label column in the dataset"
    })
    predictions: Path = field(default=None, metadata={
        "help": "Path to write outputs with MIA results"
    })

@dataclass
class MiaArguments:
    mia_method: str = field(default='all', metadata={
        "help": "MIA methodology"
    })

@dataclass
class Arguments:
    mia: MiaArguments
    data: DataArguments

def main(args: Arguments):

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Load data for inference
    inference_data = datasets.load_from_disk(str(args.data.inference_path))

    # Load synthetic data
    synthetic_data = datasets.Dataset.from_json(str(args.data.synthetic_path))

    # Compute MIA score - right now let's compute it all
    scores, _ = compute_mia_score(inference_data[args.data.text_name], synthetic_data[args.data.text_name], method='all')
    
    # add column with scores
    for method in scores.keys():
        inference_data = inference_data.add_column(f"mi_signal_{method}", scores[method])

    inference_data.save_to_disk(args.data.predictions)

if __name__ == "__main__":
    arg_parser = transformers.HfArgumentParser(
        (DataArguments, MiaArguments)
    )
    data_args, mia_args = arg_parser.parse_args_into_dataclasses()
    main(Arguments(data=data_args, mia=mia_args))

