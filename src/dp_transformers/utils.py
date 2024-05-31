import json
from dataclasses import dataclass, asdict
from typing import List
from pathlib import Path


@dataclass
class TrainingMetadata:
    hf_base_model: str
    model_history: List[str]

    def save_pretrained(self, path: Path):
        path = Path(path)
        if path.is_dir():
            path = path/"training_metadata.json"
        assert path.suffix == ".json", "Path should be a json file"
        with open(path, "w") as f:
            json.dump(asdict(self), f)

    @classmethod 
    def from_pretrained(cls, model_name_or_path: str, require_path: bool = False):
        path = Path(model_name_or_path)
        if not path.exists() and not require_path:
            # Assume we're using a model_name and not a path
            return TrainingMetadata(hf_base_model=model_name_or_path, model_history=[model_name_or_path])

        if path.is_dir():
            path = path/"training_metadata.json"

        assert path.suffix == ".json", "Path should be a json file"
        with open(path, "r") as f:
            return cls(**json.load(f))
