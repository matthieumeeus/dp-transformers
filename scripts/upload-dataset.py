from dataclasses import dataclass
from typing import Optional
from transformers import HfArgumentParser
from datasets import load_dataset, Dataset
from abc import ABC, abstractmethod
from tempfile import TemporaryDirectory
from pathlib import Path
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

from utils import AmlArguments, make_name_aml_safe


@dataclass
class DataArguments:
    dataset_name: str


@dataclass
class Arguments:
    data: DataArguments
    aml: AmlArguments


class DatasetLoader(ABC):
    @abstractmethod
    def load_dataset(self) -> Dataset:
        pass

    @abstractmethod
    def description(self) -> str:
        pass    


class RedditJsonlPromptCompletion(DatasetLoader):
    def load_dataset(self) -> Dataset:
        dataset = load_dataset("reddit", split="train")

        dataset = dataset.select_columns(["content"]).rename_column("content", "completion")
        dataset = dataset.add_column("prompt", ["" for _ in range(len(dataset))])

        return dataset
    
    def description(self) -> str:
        return (
            "Reddit dataset from HF hub in jsonl format. "
            "Original dataset's content column is renamed to completion and prompt is added with empty strings"
        )


class RedditJsonlPromptCompletion100kTrain(DatasetLoader):
    def load_dataset(self) -> Dataset:
        dataset = load_dataset("reddit", split="train[:100000]")

        dataset = dataset.select_columns(["content"]).rename_column("content", "completion")
        dataset = dataset.add_column("prompt", ["" for _ in range(len(dataset))])

        return dataset
    
    def description(self) -> str:
        return (
            "First 100k samples from Reddit dataset from HF hub in jsonl format. "
            "Original dataset's content column is renamed to completion and prompt is added with empty strings"
        )
    

class RedditJsonlPromptCompletion100kValidation(DatasetLoader):
    def load_dataset(self) -> Dataset:
        dataset = load_dataset("reddit", split="train[100000:120000]")

        dataset = dataset.select_columns(["content"]).rename_column("content", "completion")
        dataset = dataset.add_column("prompt", ["" for _ in range(len(dataset))])

        return dataset
    
    def description(self) -> str:
        return (
            "Samples 100k - 120k from Reddit dataset from HF hub in jsonl format. "
            "Original dataset's content column is renamed to completion and prompt is added with empty strings"
        )


class RedditJsonlPromptCompletion500kTrain(DatasetLoader):
    def load_dataset(self) -> Dataset:
        dataset = load_dataset("reddit", split="train[:500000]")

        dataset = dataset.select_columns(["content"]).rename_column("content", "completion")
        dataset = dataset.add_column("prompt", ["" for _ in range(len(dataset))])

        return dataset
    
    def description(self) -> str:
        return (
            "First 500k samples from Reddit dataset from HF hub in jsonl format. "
            "Original dataset's content column is renamed to completion and prompt is added with empty strings"
        )

class RedditJsonlPromptCompletion500kValidation(DatasetLoader):
    def load_dataset(self) -> Dataset:
        dataset = load_dataset("reddit", split="train[500000:600000]")

        dataset = dataset.select_columns(["content"]).rename_column("content", "completion")
        dataset = dataset.add_column("prompt", ["" for _ in range(len(dataset))])

        return dataset
    
    def description(self) -> str:
        return (
            "Samples 500k - 600k from Reddit dataset from HF hub in jsonl format. "
            "Original dataset's content column is renamed to completion and prompt is added with empty strings"
        )
    

class HelpSteerTrain(DatasetLoader):
    def load_dataset(self) -> Dataset:
        dataset = load_dataset("nvidia/HelpSteer", split="train")

        dataset = dataset.select_columns(["prompt", "response"]).rename_column("response", "completion")

        return dataset
    
    def description(self) -> str:
        return (
            "HelpSteer dataset from HF hub. https://huggingface.co/datasets/nvidia/HelpSteer"
        )
    
class HelpSteerValidation(DatasetLoader):
    def load_dataset(self) -> Dataset:
        dataset = load_dataset("nvidia/HelpSteer", split="validation")

        dataset = dataset.select_columns(["prompt", "response"]).rename_column("response", "completion")

        return dataset
    
    def description(self) -> str:
        return (
            "HelpSteer dataset from HF hub. https://huggingface.co/datasets/nvidia/HelpSteer"
        )


# Add custom dataset loaders here. See RedditTrain for an example


DATASET_LOADERS = {cls.__name__: cls for cls in DatasetLoader.__subclasses__()}


def main(args: Arguments):
    if args.data.dataset_name not in DATASET_LOADERS:
        raise ValueError(f"Dataset {args.data.dataset_name} not found. Available datasets: {list(DATASET_LOADERS.keys())}")

    dataset_loader = DATASET_LOADERS[args.data.dataset_name]()
    dataset = dataset_loader.load_dataset()

    if {"prompt", "completion"} != set(dataset.column_names):
        raise ValueError(f"Dataset should have columns 'prompt' and 'completion' but has {dataset.column_names}")

    ml_client = args.aml.ml_client

    with TemporaryDirectory() as temp_dir:
        temp_file = Path(temp_dir)/"dataset.json"
        dataset.to_json(temp_file)

        aml_dataset_name = make_name_aml_safe(args.data.dataset_name)

        data_asset = Data(
            name=aml_dataset_name,
            description=dataset_loader.description(),
            path=temp_file,
            type=AssetTypes.URI_FILE
        )

        data_asset = ml_client.data.create_or_update(data_asset)

    print(f"Dataset {args.data.dataset_name} uploaded to Azure ML workspace {ml_client.workspace_name}")
    print(f"See https://ml.azure.com/data/{data_asset.name}/{data_asset.version}/details?wsid=/"
          f"subscriptions/{ml_client.subscription_id}/"
          f"resourceGroups/{ml_client.resource_group_name}/"
          f"providers/Microsoft.MachineLearningServices/workspaces/{ml_client.workspace_name}")


if __name__ == "__main__":
    parser = HfArgumentParser((DataArguments, AmlArguments))
    data_args, aml_args = parser.parse_args_into_dataclasses()
    main(Arguments(data=data_args, aml=aml_args))
