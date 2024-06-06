from pydantic import BaseModel
from pydantic_cli import run_and_exit
from pathlib import Path
from datasets import load_from_disk

class Arguments(BaseModel):
    all_data: Path
    min_words: int
    text_column: str
    filtered_data: Path

def select_valid_sentences(dataset, text_name, min_num_words):

    # first get the valid indices based on the min_num_words
    valid_indices = []
    for idx in range(len(dataset)):
        sample = dataset[idx][text_name]
        if len(sample.split()) >= min_num_words:
            valid_indices.append(idx)
    
    valid_sub_dataset = dataset.select(valid_indices)
    print(f"Filtered dataset from {len(dataset)} to {len(valid_sub_dataset)} samples.")
    return valid_sub_dataset
  
def main(args: Arguments) -> int:

    # load train dataset
    dataset = load_from_disk(args.all_data, keep_in_memory=True)

    # filter on min words
    valid_dataset = select_valid_sentences(dataset, args.text_column, args.min_words)

    # save the datasets
    valid_dataset.save_to_disk(args.filtered_data)
    
    return 0

if __name__ == "__main__":
    run_and_exit(Arguments, main)
