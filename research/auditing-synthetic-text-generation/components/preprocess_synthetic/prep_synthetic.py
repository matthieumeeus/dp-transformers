from pydantic import BaseModel
from pydantic_cli import run_and_exit
from pathlib import Path
import datasets
import os
from typing import Dict, Any, Callable
from functools import partial


class Arguments(BaseModel):
    synthetic_data_path: Path
    real_label_name: str
    real_text_name: str
    synthetic_label_name: str
    synthetic_text_name: str
    prep_synthetic_data_path: Path

def main(args: Arguments) -> int:

    # load the synthetic data
    path_to_csvs = [file for file in os.listdir(args.synthetic_data_path) if file.endswith('.csv')]
    all_datasets = []
    for path in path_to_csvs:
        dataset = datasets.load_dataset('csv', data_files={'train':os.path.join(args.synthetic_data_path, path)})
        all_datasets.append(dataset['train'])
    synthetic_dataset = datasets.concatenate_datasets(all_datasets)
    
    prep_synthetic_dataset = synthetic_dataset.map(lambda x: {args.real_text_name: x[args.synthetic_text_name],
                                                              args.real_label_name: x[args.synthetic_label_name]}, 
                                    remove_columns=[args.synthetic_label_name, args.synthetic_text_name])

    prep_synthetic_dataset.to_json(args.prep_synthetic_data_path)
    
    return 0

if __name__ == "__main__":
    run_and_exit(Arguments, main)
