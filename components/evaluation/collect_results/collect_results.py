import pandas as pd
import json
import mlflow
from pydantic_cli import run_and_exit
from pydantic import BaseModel
from pathlib import Path


class Arguments(BaseModel):
    mt_bench: Path
    lm_eval: Path
    results: Path


def read_mt_bench(path: Path) -> pd.DataFrame:
    mt_bench = pd.read_json(path, lines=True)

    mt_bench = mt_bench.melt(id_vars=["category", "turn"], var_name="name", value_name="value")
    mt_bench["task"] = mt_bench.apply(lambda row: f"mt-bench/{row['category']}/{row['turn']}", axis=1)
    mt_bench = mt_bench.drop(columns=["category", "turn"])

    return mt_bench


def read_lm_eval(path: Path) -> pd.DataFrame:
    with (path/"results.json").open("r") as f:
        raw_results = json.load(f)
    lm_eval = pd.DataFrame()
    for name, values in raw_results["results"].items():
        for metric, value in values.items():
            if metric == "alias":
                continue
            lm_eval = pd.concat([
                lm_eval, pd.DataFrame({"task": [f"lm-eval/{name}"], "name": [metric], "value": [value]})
            ], axis=0)
    if "groups" in raw_results:
        for name, values in raw_results["groups"].items():
            for metric, value in values.items():
                if metric == "alias":
                    continue
                lm_eval = pd.concat([
                    lm_eval, pd.DataFrame({"task": [f"lm-eval/{name}"], "name": [metric], "value": [value]})
                ], axis=0)

    return lm_eval


def log_results_to_aml(results: pd.DataFrame) -> None:
    for _, row in results.iterrows():
        log_operation = mlflow.log_metric(row["task"] + "/" + row["name"], row["value"], synchronous=True)
    if log_operation is not None:
        log_operation.wait()


def main(args: Arguments) -> int:
    results = pd.DataFrame()

    if args.mt_bench is not None:
        mt_bench = read_mt_bench(args.mt_bench)
        results = pd.concat([results, mt_bench], axis=0)

    if args.lm_eval is not None:
        lm_eval = read_lm_eval(args.lm_eval)
        results = pd.concat([results, lm_eval], axis=0)

    log_results_to_aml(results)

    results.to_json(args.results, lines=True, orient="records")

    return 0


def exception_handler(ex):
    raise RuntimeError("An error occurred while running the script.") from ex

if __name__ == "__main__":
    run_and_exit(Arguments, main, exception_handler=exception_handler)
