import numpy as np
from pydantic import BaseModel
from pydantic_cli import run_and_exit
from pathlib import Path
from datasets import load_from_disk

class Arguments(BaseModel):
    train_data: Path
    text_name: str
    canaries_min_words: int
    n_canaries: int
    seed: int
    updated_training_data: Path
    canary_data: Path
  
def main(args: Arguments) -> int:

    # load train dataset
    train_dataset = load_from_disk(args.train_data, keep_in_memory=True)

    # first get the valid indices based on the min_num_words
    valid_indices = []
    for idx in range(len(train_dataset)):
        sample = train_dataset[idx][args.text_name]
        if len(sample.split()) >= args.canaries_min_words:
            valid_indices.append(idx)

    print(f"Number of valid samples: {len(valid_indices)}")

    # select the canary indices
    if len(valid_indices) < args.n_canaries:
        raise ValueError(f"Cannot select {args.n_canaries} canaries from {len(valid_indices)} samples.")
    
    canary_indices = np.random.choice(valid_indices, args.n_canaries, replace=False)
    non_canary_indices = [idx for idx in range(len(train_dataset)) if idx not in canary_indices]
    
    canary_data = train_dataset.select(canary_indices)

    # make sure all canaries have the same number of max words
    def truncate_sample(record):
        sample_split = record[args.text_name].split()
        truncated_sample = " ".join(sample_split[:args.canaries_min_words])
        record[args.text_name] = truncated_sample
        return record

    canary_data = canary_data.map(truncate_sample)

    updated_training_data = train_dataset.select(non_canary_indices)

    # save the datasets
    canary_data.save_to_disk(args.canary_data)
    updated_training_data.save_to_disk(args.updated_training_data)
    
    return 0

if __name__ == "__main__":
    run_and_exit(Arguments, main)
