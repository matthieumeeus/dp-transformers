from pydantic import BaseModel
from pydantic_cli import run_and_exit
from pathlib import Path
import numpy as np
import datasets


class Arguments(BaseModel):
    all_predictions: Path
    mia_method: str
    selected_predictions: Path

def main(args: Arguments) -> int:

    all_predictions = datasets.load_from_disk(str(args.all_predictions))
    print(all_predictions)
    # if we have the ngram likelihood, add the dignal as log signal and the exponent as normal mi signal
    if 'ngram' in args.mia_method:
        all_predictions = all_predictions.rename_column(f"mi_signal_{args.mia_method}", "log_mi_signal")
        all_predictions = all_predictions.add_column("mi_signal", [np.exp(k).astype(np.double) for k in all_predictions["log_mi_signal"]])
    # if not, we just rename the column
    else:
        all_predictions = all_predictions.rename_column(f"mi_signal_{args.mia_method}", "mi_signal")
    all_predictions.save_to_disk(args.selected_predictions)

    return 0

if __name__ == "__main__":
    run_and_exit(Arguments, main)
