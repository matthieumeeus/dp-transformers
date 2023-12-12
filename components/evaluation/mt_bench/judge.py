import os
import gen_judgment
import pandas as pd

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
    judgments: Path
    results: Path
    answers: Path
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


def analyze_results_single(judgments: pd.DataFrame, questions: pd.DataFrame) -> pd.DataFrame:
    questions = questions[["question_id", "category"]]

    assert len(set(judgments["model"])) == 1, "Multiple models in judgments"
    judgments = judgments[["question_id", "score", "judge"]]

    print(f"Found {len(judgments)} judgments")
    print(f"Found {len(questions)} questions")

    results = judgments.merge(questions, on="question_id", how="inner")
    results["judge_model"], results["judge_prompt"] = zip(*results["judge"])
    results.drop(columns=["judge"], inplace=True)

    if len(results[(0 > results["score"]) | (results["score"] > 10)]) > 0:
        print("Some judgment scores outside of [0, 10] range:")
        print(results[(0 > results['score']) | (results['score'] > 10)])
        print("Ignoring these scores...")

        results = results[(0 <= results["score"]) & (results["score"] <= 10)]

    if len(results) != len(results[["question_id", "judge_model", "judge_prompt"]].drop_duplicates()):
        raise ValueError(f"Duplicate judgments: {results[results.duplicated(subset=['question_id', 'judge_model', 'judge_prompt'], keep=False)]}")


    results_all = results.groupby(["judge_model", "judge_prompt"]).agg({"score": ["mean", "std", "count", "min", "max"]}).reset_index()
    results_all["category"] = "all"

    results = results.groupby(["category", "judge_model", "judge_prompt"]).agg({"score": ["mean", "std", "count", "min", "max"]}).reset_index()
    results = pd.concat([results, results_all], axis=0)

    # flatten indices
    results.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in results.columns.values]

    return results



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

    print("", flush=True)

    if args.judgments.exists():
        print(f"Judgments already exist at {args.judgments}, skipping generation")
    else:
        print("Generating judgments...")
        proc = Popen(gen_judgment_call, stdin=PIPE, env=env)
        # Script asks for 'enter' to continue, simulate this input here
        proc.communicate(input=b"\n")
        if proc.returncode != 0:
            raise CalledProcessError(proc.returncode, gen_judgment_script)

        copy2(cwd/"data"/"mt_bench"/"model_judgment"/"gpt-4_single.jsonl", args.judgments)

    results = analyze_results_single(
        judgments=pd.read_json(args.judgments, lines=True),
        questions=pd.read_json(cwd/"data"/"mt_bench"/"question.jsonl", lines=True)
    )

    run.log_table("results", results.to_dict(orient="list"))

    results.to_json(args.results, orient="records", lines=True)

    return 0


def exception_handler(ex):
    raise RuntimeError("Error in evaluation") from ex


if __name__ == "__main__":
    run_and_exit(Arguments, main, exception_handler=exception_handler)

