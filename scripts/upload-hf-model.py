import os
import transformers

from dataclasses import dataclass
from transformers import HfArgumentParser, AutoModelForCausalLM, AutoTokenizer
from tempfile import TemporaryDirectory
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from typing import Optional

from dp_transformers.utils import TrainingMetadata
from utils import AmlArguments, make_name_aml_safe


@dataclass
class ModelArguments:
    model_name: str
    trust_remote_code: bool = False
    hf_token: Optional[str] = None


@dataclass
class Arguments:
    model: ModelArguments
    aml: AmlArguments



def main(args: Arguments):
    if os.path.isdir(args.model.model_name):
        raise ValueError("Model name should be a model on HF hub. Received a directory instead.")

    optional_hf_args = {"token": args.model.hf_token} if args.model.hf_token else {}

    model = AutoModelForCausalLM.from_pretrained(args.model.model_name, trust_remote_code=args.model.trust_remote_code, **optional_hf_args)
    tokenizer = AutoTokenizer.from_pretrained(args.model.model_name, trust_remote_code=args.model.trust_remote_code, **optional_hf_args)
    training_metadata = TrainingMetadata.from_pretrained(args.model.model_name)

    ml_client = args.aml.ml_client

    with TemporaryDirectory() as tmp_dir:
        model.save_pretrained(tmp_dir)
        training_metadata.save_pretrained(tmp_dir)
        tokenizer.save_pretrained(tmp_dir)

        aml_model_name = make_name_aml_safe(args.model.model_name)

        model_asset = Data(
            name=aml_model_name,
            description=f"Model {args.model.model_name} from HF hub using transformers version: {transformers.__version__}",
            path=tmp_dir,
            type=AssetTypes.URI_FOLDER
        )

        model_asset = ml_client.data.create_or_update(model_asset)

    print(f"Model {args.model.model_name} uploaded to Azure ML workspace {ml_client.workspace_name}")
    print(f"See https://ml.azure.com/data/{model_asset.name}/{model_asset.version}/details?wsid=/"
          f"subscriptions/{ml_client.subscription_id}/"
          f"resourceGroups/{ml_client.resource_group_name}/"
          f"providers/Microsoft.MachineLearningServices/workspaces/{ml_client.workspace_name}")


if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, AmlArguments))
    model_args, aml_args = parser.parse_args_into_dataclasses()
    main(Arguments(model=model_args, aml=aml_args))