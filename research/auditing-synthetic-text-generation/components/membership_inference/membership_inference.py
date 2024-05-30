
'''Trained roberta sentiment classifier on the synthetic data to compute utility'''

from pathlib import Path
import numpy as np
import transformers
import datasets
import sys
import json
import logging
from mia_methods import compute_mia_score
from mia_utils import select_samples, compute_performance
import random

from dataclasses import dataclass, field
from typing import Optional, Union

logger = logging.getLogger(__name__)

@dataclass
class DataArguments:
    member_path: Path = field(default=None, metadata={
        "help": "Path to member path"
    })
    non_member_path: Path = field(default=None, metadata={
        "help": "Path to non-member path"
    })
    synthetic_path: Path = field(default=None, metadata={
        "help": "Path to synthetic path"
    })
    text_name: str = field(default="sentence", metadata={
        "help": "Name of the text column in the dataset"
    })
    label_name: str = field(default="label", metadata={
        "help": "Name of the label column in the dataset"
    })
    number_of_samples: int = field(default=1000, metadata={
        "help": "Number of samples for MIA"
    })
    seed: int = field(default=42, metadata={
        "help": "Random seed"
    })
    mia_results_path: Path = field(default=None, metadata={
        "help": "Path to write outputs with MIA results"
    })

@dataclass
class MiaArguments:
    n_runs: int = field(default=1, metadata={
        "help": "Number of runs"
    })
    method: str = field(default='all', metadata={
        "help": "MIA methodology"
    })

@dataclass
class Arguments:
    mia: MiaArguments
    data: DataArguments

def main(args: Arguments):
    random.seed(args.data.seed)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Load data
    # start with the members and non members
    all_members = datasets.Dataset.from_json(str(args.data.member_path))
    all_non_members = datasets.Dataset.from_json(str(args.data.non_member_path))

    mia_performance_across_runs = []
    for run in range(args.mia.n_runs):
        logger.info(f"Run {run+1}/{args.mia.n_runs}")
        members_selected = select_samples(all_members, args.data.text_name, args.data.number_of_samples)
        non_members_selected = select_samples(all_non_members, args.data.text_name, args.data.number_of_samples)

        # Load the synthetic data
        synthetic_data = datasets.Dataset.from_json(str(args.data.synthetic_path))[args.data.text_name]

        # Compute MIA scores
        if run == 0:
            scores_members, synthetic_embeddings = compute_mia_score(members_selected, synthetic_data, method=args.mia.method)
        else:
            scores_members, _ = compute_mia_score(members_selected, synthetic_data, method=args.mia.method, synthetic_embeddings=synthetic_embeddings)
        scores_non_members, _ = compute_mia_score(non_members_selected, synthetic_data, method=args.mia.method, synthetic_embeddings=synthetic_embeddings)

        # Compute MIA performance
        mia_performance = compute_performance(scores_members, scores_non_members)
        logger.info(f"MIA performance: {mia_performance}")
        mia_performance_across_runs.append(mia_performance)
    
    # print out the aggregate
    for method in mia_performance_across_runs[0].keys():
        print(f"Method: {method}")
        auc_vals = [p[method]['auc'] for p in mia_performance_across_runs]
        print(f"AUC: {np.mean(auc_vals)} +/- {np.std(auc_vals)}")
        tpr_001_vals = [p[method]['tpr_at_0.01'] for p in mia_performance_across_runs]
        print(f"TPR@0.01: {np.mean(tpr_001_vals)} +/- {np.std(tpr_001_vals)}")
        print('---')

    # Save the MIA performance
    with open(args.data.mia_results_path, "w") as f:
        json.dump(mia_performance_across_runs, f)

if __name__ == "__main__":
    arg_parser = transformers.HfArgumentParser(
        (DataArguments, MiaArguments)
    )
    data_args, mia_args = arg_parser.parse_args_into_dataclasses()
    main(Arguments(data=data_args, mia=mia_args))

