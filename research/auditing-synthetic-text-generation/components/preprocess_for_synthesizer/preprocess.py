from pydantic import BaseModel
from pydantic_cli import run_and_exit
from pathlib import Path
from datasets import load_from_disk
from typing import Dict, Any, Callable
from functools import partial


class Arguments(BaseModel):
    dataset: Path
    train_data: Path
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
    dataset = load_from_disk(args.dataset, keep_in_memory=True)
    dataset = dataset.map(
        partial(
            convert_classification_record_to_synthesizer_record,
            label_int2str=dataset.features["label"].int2str,
            text_name=args.text_name,
            templated_prompt=args.templated_prompt,
            label_name=args.label_name
        ),
        remove_columns=dataset.column_names
    )

    dataset.to_json(args.train_data)
    
    return 0


if __name__ == "__main__":
    run_and_exit(Arguments, main)
