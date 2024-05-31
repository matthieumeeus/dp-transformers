from pydantic_cli import run_and_exit
from pydantic import BaseModel
from datasets import load_from_disk
from typing import Optional


class Arguments(BaseModel):
    hf_dataset_path: str
    output_path: str
    completion_column: str
    prompt_column: Optional[str] = None


def main(args: Arguments) -> int:
    dataset = load_from_disk(args.hf_dataset_path)

    if args.prompt_column is None:
        prompt_column = "prompt"
        dataset = dataset.add_column(prompt_column, ["" for _ in range(len(dataset))])
    else:
        prompt_column = args.prompt_column

    dataset = dataset.rename_columns({args.completion_column: "completion", prompt_column: "prompt"})
    dataset = dataset.select_columns(["prompt", "completion"])

    dataset.to_json(args.output_path)

    return 0


def exception_handler(exception: Exception) -> int:
    raise RuntimeError("An exception occurred") from exception


if __name__ == "__main__":
    run_and_exit(Arguments, main, exception_handler=exception_handler)
