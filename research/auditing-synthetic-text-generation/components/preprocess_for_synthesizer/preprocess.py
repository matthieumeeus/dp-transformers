from pydantic import BaseModel
from pydantic_cli import run_and_exit
from pathlib import Path
import datasets
from typing import Dict, Any, Callable
from functools import partial


class Arguments(BaseModel):
    train_data_path: Path
    fine_tune_train_data: Path
    eval_data_path: Path
    fine_tune_eval_data: Path
    templated_prompt: str
    label_name: str
    text_name: str


def convert_classification_record_to_synthesizer_record(
        record: Dict[str, Any], text_name: str, templated_prompt: str, label_name: str, label_int2str: Callable[[int], str]
    ) -> Dict[str, Any]:
    """
    Converts a classification record to a synthesizer record by replacing the label name in the templated prompt.

    Args:
        record (Dict[str, Any]): The classification record containing the text and label information.
        text_name (str): The name of the text field in the record.
        templated_prompt (str): The templated prompt with placeholders for the label name.
                                The placeholder should be in the form of '{label_name}'.
        label_name (str): The name of the label field in the record.
        label_int2str (Callable[[int], str]): A function to convert the label integer to its corresponding string representation.

    Returns:
        Dict[str, Any]: The synthesizer record with the replaced prompt and the original text.

    Raises:
        ValueError: If the label name is not found in the templated prompt.
    """
    label_str = label_int2str(record[label_name])
    if f"{{{label_name}}}" not in templated_prompt:
        raise ValueError(f"Label name '{{{label_name}}}' not found in templated prompt: {templated_prompt}")
    prompt = templated_prompt.replace(f"{{{label_name}}}", label_str)
    return {
        "prompt": prompt,
        "completion": record[text_name],
    }


def main(args: Arguments) -> int:

    # first do the eval dataset
    eval_dataset = datasets.load_from_disk(str(args.eval_data_path), keep_in_memory=True)
    int2str_mapping = eval_dataset.features["label"].int2str
    eval_dataset = eval_dataset.map(
        partial(
            convert_classification_record_to_synthesizer_record,
            label_int2str=int2str_mapping,
            text_name=args.text_name,
            templated_prompt=args.templated_prompt,
            label_name=args.label_name
        ),
        remove_columns=eval_dataset.column_names
    )
    eval_dataset.to_json(args.fine_tune_eval_data)

    # then do the train dataset
    train_dataset = datasets.load_from_disk(str(args.train_data_path), keep_in_memory=True)
    train_dataset = train_dataset.map(
        partial(
            convert_classification_record_to_synthesizer_record,
            label_int2str=int2str_mapping,
            text_name=args.text_name,
            templated_prompt=args.templated_prompt,
            label_name=args.label_name
        ),
        remove_columns=train_dataset.column_names
    )
    train_dataset.to_json(args.fine_tune_train_data)

    # then the eval dataset
    
    return 0


if __name__ == "__main__":
    run_and_exit(Arguments, main)
