from pydantic import BaseModel
from pydantic_cli import run_and_exit
from pathlib import Path
from datasets import load_from_disk
from typing import Dict, Any, Callable
from functools import partial


class Arguments(BaseModel):
    dataset: Path
    train_data: Path


def convert_classification_record_to_synthesizer_record(
        record: Dict[str, Any], label_int2str: Callable[[int], str]
    ) -> Dict[str, Any]:
    label_str = label_int2str(record["label"])
    return {
        "prompt": f"A sentence with a {label_str} sentiment: ",
        "completion": record["sentence"]
    }


def main(args: Arguments) -> int:
    dataset = load_from_disk(args.dataset)

    dataset = dataset.map(
        partial(
            convert_classification_record_to_synthesizer_record,
            label_int2str=dataset.features["label"].int2str
        ),
        remove_columns=dataset.column_names,
        num_proc=8
    )

    dataset.to_json(args.train_data)
    
    return 0


if __name__ == "__main__":
    run_and_exit(Arguments, main)
