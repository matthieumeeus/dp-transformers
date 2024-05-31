import torch
import json
import os

from shutil import copy2
from pathlib import Path
from subprocess import check_call
from pydantic_cli import run_and_exit
from pydantic import BaseModel
from urllib.request import urlretrieve
from fastchat.llm_judge import gen_model_answer
from tempfile import TemporaryDirectory
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import get_fschat_version 


class Arguments(BaseModel):
    model: Path
    num_gpus_per_model: int
    answers: Path
    used_model_id: Path
    trust_remote_code: str
    num_total_gpus: int = torch.cuda.device_count()
    peft: Path = None
    model_id: str = None


def download_llm_judge_data(fschat_version: str, download_path: Path):
    mt_bench_data_path = download_path/"data"/"mt_bench"
    mt_bench_data_path.mkdir(parents=True, exist_ok=True)
    urlretrieve(f"https://raw.githubusercontent.com/lm-sys/FastChat/{fschat_version}/fastchat/llm_judge/data/mt_bench/question.jsonl", mt_bench_data_path/"question.jsonl")


def main(args: Arguments) -> int:
    cwd = Path.cwd()

    fschat_version = get_fschat_version()
    print(f"Found FastChat version {fschat_version}")
    download_llm_judge_data(fschat_version=fschat_version, download_path=cwd)


    with TemporaryDirectory() as merged_model_dir:
        if args.peft is None:
            model = args.model
        else:
            trust_remote_code = bool(json.loads(args.trust_remote_code.lower()))
            # Merge peft model and base model
            PeftModel.from_pretrained(
                AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=trust_remote_code),
                args.peft
            ).merge_and_unload().save_pretrained(merged_model_dir)
            AutoTokenizer.from_pretrained(args.model).save_pretrained(merged_model_dir)
            model = Path(merged_model_dir)

        if args.model_id is not None:
            model_id = args.model_id
            print("Using command line provided model ID. Model ID from model directory will be ignored.")
        else:
            print("No model ID provided, attempting to read from model directory...")
            model_id_path = args.model/"training_metadata.json"
            with (model_id_path).open("r") as f:
                model_id = os.path.basename(json.load(f)["hf_base_model"])

        print(f"Using model ID: {model_id}")
        with args.used_model_id.open("w") as f:
            f.write(model_id)

        gen_answers_script = gen_model_answer.__file__
        gen_answers_call = [
            "python", gen_answers_script, "--model-path", str(model), "--model-id", model_id,
            "--num-gpus-total", str(args.num_total_gpus), "--num-gpus-per-model", str(args.num_gpus_per_model)
        ]

        print(" ".join(gen_answers_call))
        check_call(gen_answers_call)
        copy2(cwd/"data"/"mt_bench"/"model_answer"/(model_id + ".jsonl"), args.answers)

    return 0


def exception_handler(ex):
    raise RuntimeError("Error in evaluation") from ex


if __name__ == "__main__":
    run_and_exit(Arguments, main, exception_handler=exception_handler)
