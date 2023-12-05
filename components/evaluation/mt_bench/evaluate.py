import torch
from shutil import copy2
from pathlib import Path
from subprocess import check_call
from pydantic_cli import run_and_exit
from pydantic import BaseModel
from urllib.request import urlretrieve

from fastchat.llm_judge import gen_model_answer, gen_judgment


class Arguments(BaseModel):
    model: Path
    num_gpus_per_model: int
    num_concurrent_api_calls: int
    results: Path
    judgements: Path
    answers: Path
    num_total_gpus: int = torch.cuda.device_count()


def main(args: Arguments) -> int:
    # Download data
    mt_bench_data_path = Path("data")/"mt_bench"
    mt_bench_data_path.mkdir(parents=True, exist_ok=True)
    urlretrieve("https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/question.jsonl", mt_bench_data_path/"question.jsonl")

    model_id = "model"

    gen_answers_script = gen_model_answer.__file__
    gen_answers_call = [
        "python", gen_answers_script, "--model-path", str(args.model), "--model-id", model_id,
        "--num-gpus-total", str(args.num_total_gpus), "--num-gpus-per-model", str(args.num_gpus_per_model)
    ]
    print(" ".join(gen_answers_call))
    check_call(gen_answers_call)
    copy2(Path("data")/"mt_bench"/"model_answer"/(model_id+".jsonl"), args.answers)

    gen_judgment_script = gen_judgment.__file__
    gen_judgment_call = [
        "python", gen_judgment_script, "--model-id", str(model_id), "--parallel", str(args.num_concurrent_api_calls)
    ]
    print(" ".join(gen_judgment_call))
    check_call(gen_judgment_call)
    copy2(Path("data")/"mt_bench"/"model_judgement"/(model_id+".jsonl"), args.judgements)

    return 0


def exception_handler(ex):
    raise RuntimeError("Error in evaluation") from ex


if __name__ == "__main__":
    run_and_exit(Arguments, main, exception_handler=exception_handler)
