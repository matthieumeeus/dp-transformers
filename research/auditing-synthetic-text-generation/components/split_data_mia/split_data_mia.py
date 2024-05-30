from pydantic import BaseModel
from pydantic_cli import run_and_exit
from pathlib import Path
from datasets import load_from_disk
import random

class Arguments(BaseModel):
    all_train_data_path: Path
    min_words: int
    n_hold_out: int
    text_column: str
    member_data: Path
    non_member_data: Path
    seed: int

def select_valid_sentences(dataset, text_name, min_num_words):

    # first get the valid indices based on the min_num_words
    valid_indices = []
    for idx in range(len(dataset)):
        sample = dataset[idx][text_name]
        if len(sample.split()) >= min_num_words:
            valid_indices.append(idx)
    
    valid_sub_dataset = dataset.select(valid_indices)
    return valid_sub_dataset
  
def main(args: Arguments) -> int:
    # set the seed
    random.seed(args.seed)

    # load train dataset
    train_dataset = load_from_disk(args.all_train_data_path, keep_in_memory=True)

    # filter on min words
    valid_train_dataset = select_valid_sentences(train_dataset, args.text_column, args.min_words)

    # split into member and non-member
    dataset_split = valid_train_dataset.train_test_split(test_size=args.n_hold_out, seed=args.seed)
    member_data = dataset_split['train']
    non_member_data = dataset_split['test']

    # save the datasets
    print(f"Holding out {args.n_hold_out} samples for non-member data - leaving {len(member_data)} samples for member data.")
    member_data.to_json(args.member_data)
    non_member_data.to_json(args.non_member_data)
    
    return 0

if __name__ == "__main__":
    run_and_exit(Arguments, main)
