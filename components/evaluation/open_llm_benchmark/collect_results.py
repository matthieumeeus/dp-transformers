import json
from pydantic import BaseModel
from pydantic_cli import run_and_exit
from pathlib import Path
from typing import Dict


class Arguments(BaseModel):
    arc_challenge: Path
    hellaswag: Path
    truthfulqa_mc2: Path
    mmlu: Path
    winogrande: Path
    gsm8k: Path
    results: Path


def append_lm_eval_results(results: Dict[str, Dict], result: Dict[str, Dict]) -> Dict[str, Dict]:
    assert len(set(result["results"].keys()).intersection(results["results"].keys())) == 0
    results["results"].update(result["results"])

    assert len(set(result["configs"].keys()).intersection(results["configs"].keys())) == 0
    results["configs"].update(result["configs"])

    assert len(set(result["versions"].keys()).intersection(results["versions"].keys())) == 0
    results["versions"].update(result["versions"])

    assert len(set(result["n-shot"].keys()).intersection(results["n-shot"].keys())) == 0
    results["n-shot"].update(result["n-shot"])

    if "groups" in result:
        assert len(set(result["groups"].keys()).intersection(results["groups"].keys())) == 0
        results["groups"].update(result["groups"])

    return results


def main(args: Arguments) -> int:
    results = {"results": {}, "configs": {}, "versions": {}, "n-shot": {}, "groups": {}}
    for task_dir in [args.arc_challenge, args.hellaswag, args.truthfulqa_mc2, args.mmlu, args.winogrande, args.gsm8k]:
        with (task_dir/"results.json").open("r") as f:
            result = json.load(f)
        results = append_lm_eval_results(results, result)
    with (args.results/"results.json").open("w") as f:
        json.dump(results, f, indent=2)
    return 0


def exception_handler(ex):
    raise RuntimeError("An error occurred while running the script.") from ex


if __name__ == "__main__":
    run_and_exit(Arguments, main, exception_handler=exception_handler)
