from pydantic import BaseModel
from pydantic_cli import run_and_exit
from pathlib import Path
from datasets import load_from_disk
import random
import torch
import numpy as np
from transformers import RobertaTokenizer, RobertaForMaskedLM

class Arguments(BaseModel):
    original_data: Path
    text_column: str
    num_tokens_to_replace: int
    replacement_method: str
    model_name: str
    modified_data: Path

def replace_random_tokens(sentence, tokenizer, model, num_tokens_to_replace, device: str, replacement_method='random'):

    # Tokenize the sentence, disregarding start and end of sequence
    original_tokens = np.array(tokenizer.encode(sentence)[1:-1])
    modified_tokens = original_tokens.copy()
    
    # Randomly sample `num_tokens_to_replace` token indices
    replace_indices = random.sample(range(len(original_tokens)), num_tokens_to_replace)
    
    if replacement_method == "mlm":
        for idx in replace_indices:
            # Create input ids with [MASK] tokens
            masked_token_ids = original_tokens.copy()
            masked_token_ids[idx] = tokenizer.mask_token_id
    
            # Convert to tensor and predict masked tokens
            input_ids = torch.tensor([masked_token_ids]).to(device)
            with torch.no_grad():
                outputs = model(input_ids)
                predictions = outputs.logits
    
            # Replace the selected tokens with predicted tokens
            predictions = predictions[0, idx]
            # set the proba for the current token to 0
            predictions[original_tokens[idx]] = 0
            predicted_token_id = torch.multinomial(torch.nn.functional.softmax(predictions, dim=-1), num_samples=1).item()
            modified_tokens[idx] = predicted_token_id
    elif replacement_method == "random":
        # Replace the selected tokens with random tokens from the vocabulary
        for idx in replace_indices:
            random_token_id = random.choice([k for k in tokenizer.get_vocab().values() if k != original_tokens[idx]])
            modified_tokens[idx] = random_token_id
    elif replacement_method == "all_random":
        # Replace all tokens with random tokens from the vocabulary
        for idx in range(len(original_tokens)):
            random_token_id = random.choice([k for k in tokenizer.get_vocab().values() if k != original_tokens[idx]])
            modified_tokens[idx] = random_token_id
    elif replacement_method == "all_randomx2":
        new_tokens = []
        # Replace all tokens with random tokens from the vocabulary
        for idx in range(len(original_tokens) * 2):
            random_token_id = random.choice([k for k in tokenizer.get_vocab().values()])
            new_tokens.append(random_token_id)
        modified_tokens = new_tokens
    else:
        raise ValueError(f"Invalid replacement method: {replacement_method}. Choose from ['random', 'mlm']")
    
    # Convert tokens back to sentence
    modified_sentence = tokenizer.decode(modified_tokens)

    return modified_sentence
  
def main(args: Arguments) -> int:

    # load the original dataset
    dataset = load_from_disk(args.original_data, keep_in_memory=True)

    # load model
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RobertaForMaskedLM.from_pretrained(args.model_name).to(device)

    # replace the tokens
    def process_example(example):
        modified_sentence = replace_random_tokens(example[args.text_column], tokenizer=tokenizer, model=model, 
                                                  num_tokens_to_replace=args.num_tokens_to_replace, replacement_method=args.replacement_method, device=device)
        example[args.text_column] = modified_sentence
        return example

    modified_dataset = dataset.map(process_example, batched=False)

    # save the datasets
    modified_dataset.save_to_disk(args.modified_data)
    
    return 0

if __name__ == "__main__":
    run_and_exit(Arguments, main)
