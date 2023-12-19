import pandas as pd
from tempfile import TemporaryDirectory
from judge import analyze_results_single, download_llm_judge_data
from common import get_fschat_version
from pathlib import Path


JUDGMENT_FILE = Path(__file__).parent / "judgments.json"


def test_judge():
    fschat_version = get_fschat_version()
    with TemporaryDirectory() as tmpdir:
        download_llm_judge_data(fschat_version=fschat_version, download_path=Path(tmpdir))

        judgments = pd.read_json(JUDGMENT_FILE, lines=True)
        questions = pd.read_json(Path(tmpdir)/"data"/"mt_bench"/"question.jsonl", lines=True)

        results = analyze_results_single(judgments=judgments, questions=questions)

        breakpoint()
        pass