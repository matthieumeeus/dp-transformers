from pydantic import BaseModel
from pydantic_cli import run_and_exit
from pathlib import Path
import datasets 
import sys
import torch
import logging
import random
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from canary_utils import sample_canaries_from_dataset, download_punkt_if_not_exists, make_canaries_label_compatible, \
                         get_ppl_controlled_canaries

logger = logging.getLogger(__name__)

class Arguments(BaseModel):
    original_dataset: Path
    canary_method: str
    n_canaries: int
    canary_length: int
    external_artifact: Path
    canary_text_column: str
    batch_size: int
    label_comptability_method: str
    text_column: str
    label_column: str
    seed: int
    templated_prompt: str
    min_ppl: float
    max_ppl: float
    min_temperature: float
    max_temperature: float
    canary_dataset: Path
    updated_training_dataset: Path
  
def main(args: Arguments) -> int:
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    print("args: ", args)
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    download_punkt_if_not_exists()
    if torch.cuda.is_available():
        device = torch.device("cuda", 0) 
    else:
        device = torch.device("cpu")

    original_dataset = datasets.load_from_disk(args.original_dataset)

    if args.canary_method == "sample_real":
        assert args.canary_text_column is not None
        external_dataset = datasets.load_from_disk(args.external_artifact)
        canaries = sample_canaries_from_dataset(dataset=external_dataset, n_canaries=args.n_canaries,
                                                canary_text_column=args.canary_text_column, canary_length=args.canary_length)

        canary_dataset, updated_training_dataset = make_canaries_label_compatible(canaries=canaries, original_dataset=original_dataset, 
                                                        label_comptability_method=args.label_comptability_method,
                                                        text_name=args.text_column, label_name=args.label_column)
    elif args.canary_method == "sample_synthetic":
        tokenizer = AutoTokenizer.from_pretrained(args.external_artifact)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model = AutoModelForCausalLM.from_pretrained(args.external_artifact).to(device)
        canary_dataset, updated_training_dataset = get_ppl_controlled_canaries(original_dataset=original_dataset, label_comptability_method=args.label_comptability_method, 
                                text_name=args.text_column, label_name=args.label_column,
                                model=model, tokenizer=tokenizer, 
                                n_canaries=args.n_canaries, canary_length=args.canary_length,
                                templated_prompt=args.templated_prompt, min_ppl=args.min_ppl, max_ppl=args.max_ppl,
                                min_temperature=args.min_temperature, max_temperature=args.max_temperature,
                                batch_size=args.batch_size, device=device)

    # save the datasets
    canary_dataset.save_to_disk(args.canary_dataset)
    updated_training_dataset.save_to_disk(args.updated_training_dataset)
    
    return 0

if __name__ == "__main__":
    run_and_exit(Arguments, main)
