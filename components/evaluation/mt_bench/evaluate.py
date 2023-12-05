from shutil import copy2
from pathlib import Path
from subprocess import check_call
from pydantic_cli import run_and_exit
from pydantic import BaseModel

from fastchatllm_judge import gen_model_answer, gen_judgement, show_result


class Arguments(BaseModel):
    model: str
    model_args: str = ""
    tasks: str = None
    provide_description: bool = False
    num_fewshot: int = 0
    batch_size: str = None
    max_batch_size: int = None
    device: str = None
    output_path: Path = None
    limit: float = None
    data_sampling: float = None
    no_cache: bool = False
    decontamination_ngrams_path: str = None
    description_dict_path: Path = None
    check_integrity: bool = False
    write_out: bool = False
    output_base_path: str = None


def main(args: Arguments) -> int:
    gen_answers_script = gen_model_answer.__file__
    model_id = model_path.name
    check_call(["python", gen_answers_script, "--model-path", str(model_path), "--model-id", model_id,
                "--num-gpus-total", num_gpus_total, "--num-gpus-per-model", num_gpus_per_model])

    gen_judgement_script = gen_judgement.__file__
    check_call(["python", gen_judgement_script, "--model-id", model_id, "--parallel", num_concurrent_api_calls])

    show_result_script = show_result.__file__
    check_call(["python", show_result_script, "--model-id", model_id, "--parallel", num_concurrent_api_calls])

    return 0


if __name__ == "__main__":
    run_and_exit(main, Arguments)
