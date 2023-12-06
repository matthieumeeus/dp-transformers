import torch
import fastchat
import os

from shutil import copy2
from pathlib import Path
from subprocess import check_call, Popen, PIPE, CalledProcessError
from pydantic_cli import run_and_exit
from pydantic import BaseModel
from urllib.request import urlretrieve
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient

from fastchat.llm_judge import gen_model_answer, gen_judgment


class Arguments(BaseModel):
    model: Path
    num_gpus_per_model: int
    num_concurrent_api_calls: int
    results: Path
    judgements: Path
    answers: Path
    openai_api_base: str = None
    openai_api_key_secret_name: str = None
    openai_api_type: str = None
    openai_api_version: str = None
    vault_uri: str = None
    num_total_gpus: int = torch.cuda.device_count()


def get_fschat_version() -> str:
    return f"v{fastchat.__version__}"


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

    aoai_env = dict()
    if args.openai_api_base is not None:
        aoai_env["OPENAI_API_BASE"] = args.openai_api_base
    if args.openai_api_key_vault_name is not None and args.vault_uri is not None:
        credential = DefaultAzureCredential()
        secret_client = SecretClient(credential=credential, vault_url=args.vault_uri)
        aoai_env["OPENAI_API_KEY"] = secret_client.get_secret(args.openai_api_key_secret_name)
    if args.openai_api_type is not None:
        aoai_env["OPENAI_API_TYPE"] = args.openai_api_type
    if args.openai_api_version is not None:
        aoai_env["OPENAI_API_VERSION"] = args.openai_api_version

    fschat_version = get_fschat_version()
    print(f"Found FastChat version {fschat_version}")
    download_llm_judge_data(fschat_version=fschat_version, download_path=cwd)

    model_id = "model"

    gen_answers_script = gen_model_answer.__file__
    gen_answers_call = [
        "python", gen_answers_script, "--model-path", str(args.model), "--model-id", model_id,
        "--num-gpus-total", str(args.num_total_gpus), "--num-gpus-per-model", str(args.num_gpus_per_model)
    ]
    print(" ".join(gen_answers_call))
    check_call(gen_answers_call)
    copy2(cwd/"data"/"mt_bench"/"model_answer"/(model_id+".jsonl"), args.answers)

    gen_judgment_script = gen_judgment.__file__
    gen_judgment_call = [
        "python", gen_judgment_script, "--model-list", str(model_id), "--parallel", str(args.num_concurrent_api_calls)
    ]
    print(" ".join(gen_judgment_call))
    env = os.environ.copy()
    env.update(aoai_env)
    proc = Popen(gen_judgment_call, stdin=PIPE, env=env)
    # Script asks for 'enter' to continue, simulate this input here
    proc.communicate(input=b"\n")
    if proc.returncode != 0:
        raise CalledProcessError(proc.returncode, gen_judgment_script)
    copy2(cwd/"data"/"mt_bench"/"model_judgement"/(model_id+".jsonl"), args.judgements)

    return 0


def exception_handler(ex):
    raise RuntimeError("Error in evaluation") from ex


if __name__ == "__main__":
    run_and_exit(Arguments, main, exception_handler=exception_handler)
