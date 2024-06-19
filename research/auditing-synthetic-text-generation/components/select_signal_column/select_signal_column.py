from pydantic import BaseModel
from pydantic_cli import run_and_exit
from pathlib import Path
import datasets


class Arguments(BaseModel):
    all_predictions: Path
    mia_method: str
    selected_predictions: Path

def main(args: Arguments) -> int:

    all_predictions = datasets.load_from_disk(str(args.all_predictions))

    # rename the right column
    all_predictions = all_predictions.rename_column(f"mi_signal_{args.mia_method}", "mi_signal")
    
    all_predictions.save_to_disk(args.selected_predictions)

    return 0

if __name__ == "__main__":
    run_and_exit(Arguments, main)
