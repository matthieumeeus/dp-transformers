import os
import gen_judgment

from shutil import copy2
from pathlib import Path
from subprocess import Popen, PIPE, CalledProcessError
from pydantic_cli import run_and_exit
from pydantic import BaseModel
from urllib.request import urlretrieve
from azureml.core import Run

from common import get_fschat_version, MODEL_ID


class Arguments(BaseModel):
    num_concurrent_api_calls: int
    results: Path
    judgements: Path
    answers: Path
    judge_model: str = "azure-gpt4"
    openai_api_base: str = None
    openai_api_key_secret_name: str = None
    openai_api_type: str = None
    openai_api_version: str = None
    benchmark: str = "mt_bench"
    judge_model: str = "gpt-4"
    baseline_model: str = "gpt-3.5-turbo"
    mode: str = "single"


def download_llm_judge_data(fschat_version: str, download_path: Path):
    # Download data
    data_path = download_path/"data"
    data_path.mkdir(parents=True, exist_ok=True)
    urlretrieve(f"https://raw.githubusercontent.com/lm-sys/FastChat/{fschat_version}/fastchat/llm_judge/data/judge_prompts.jsonl", data_path/"judge_prompts.jsonl")

    mt_bench_data_path = data_path/"mt_bench"
    mt_bench_data_path.mkdir(parents=True, exist_ok=True)
    urlretrieve(f"https://raw.githubusercontent.com/lm-sys/FastChat/{fschat_version}/fastchat/llm_judge/data/mt_bench/question.jsonl", mt_bench_data_path/"question.jsonl")

    reference_answers_path = mt_bench_data_path/"reference_answer"
    reference_answers_path.mkdir(parents=True, exist_ok=True)
    urlretrieve(f"https://raw.githubusercontent.com/lm-sys/FastChat/{fschat_version}/fastchat/llm_judge/data/mt_bench/reference_answer/gpt-4.jsonl", reference_answers_path/"gpt-4.jsonl") 


def main(args: Arguments) -> int:
    cwd = Path.cwd()

    run: Run = Run.get_context()

    aoai_env = dict()
    if args.openai_api_base is not None:
        print(f"Overriding OPENAI_API_BASE to {args.openai_api_base}")
        aoai_env["OPENAI_API_BASE"] = args.openai_api_base
    if args.openai_api_key_secret_name is not None:
        kv = run.experiment.workspace.get_default_keyvault()
        print(f"Overriding OPENAI_API_KEY to key from Azure Key Vault {args.openai_api_key_secret_name}")
        aoai_env["OPENAI_API_KEY"] = kv.get_secret(args.openai_api_key_secret_name)
    if args.openai_api_type is not None:
        print(f"Overriding OPENAI_API_TYPE to {args.openai_api_type}")
        aoai_env["OPENAI_API_TYPE"] = args.openai_api_type
    if args.openai_api_version is not None:
        print(f"Overriding OPENAI_API_VERSION to {args.openai_api_version}")
        aoai_env["OPENAI_API_VERSION"] = args.openai_api_version

    fschat_version = get_fschat_version()
    print(f"Found FastChat version {fschat_version}")
    download_llm_judge_data(fschat_version=fschat_version, download_path=cwd)

    answer_dir = cwd/"data"/"mt_bench"/"model_answer"
    answer_dir.mkdir(parents=True, exist_ok=True)
    copy2(args.answers, answer_dir/(MODEL_ID+".jsonl"))


    gen_judgment_script = gen_judgment.__file__
    gen_judgment_call = [
        "python", gen_judgment_script, "--model-list", MODEL_ID, "--parallel", str(args.num_concurrent_api_calls), "--judge-model", args.judge_model
    ]
    print(" ".join(gen_judgment_call))
    env = os.environ.copy()
    env.update(aoai_env)
    print(f"OpenAI environment:")
    print(f"OPENAI_API_BASE={env.get('OPENAI_API_BASE', '<not set>')}")
    print(f"OPENAI_API_TYPE={env.get('OPENAI_API_TYPE', '<not set>')}")
    print(f"OPENAI_API_VERSION={env.get('OPENAI_API_VERSION', '<not set>')}")
    proc = Popen(gen_judgment_call, stdin=PIPE, env=env)
    # Script asks for 'enter' to continue, simulate this input here
    proc.communicate(input=b"\n")
    if proc.returncode != 0:
        raise CalledProcessError(proc.returncode, gen_judgment_script)
    copy2(cwd/"data"/"mt_bench"/"model_judgement"/(MODEL_ID+".jsonl"), args.judgements)

    return 0


def exception_handler(ex):
    raise RuntimeError("Error in evaluation") from ex


if __name__ == "__main__":
    run_and_exit(Arguments, main, exception_handler=exception_handler)

