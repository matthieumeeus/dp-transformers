import datasets 
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from torch.nn import CrossEntropyLoss
from typing import Dict
import random
import logging
import numpy as np
import nltk
from nltk.data import find
from collections import Counter

def sample_canaries_from_dataset(dataset: datasets.Dataset, n_canaries: int,
                                 canary_text_column: str, canary_length: int):
    shuffled_indices = np.random.choice(range(len(dataset)), len(dataset), replace=False)
    canaries = []
    i = 0
    while len(canaries) < n_canaries:
        if i % 100==0:
            logging.info(f"Sampled {len(canaries)} canaries after going through {i} samples.")
            if len(canaries) > 0:
                print('The last canary was: ')
                print(canaries[-1])
        sample = dataset[int(shuffled_indices[i])]
        text = sample[canary_text_column]
        # now sample a random substring of length canary_length
        text_in_words = text.split()
        if len(text_in_words) < canary_length:
            i += 1
            continue
        else:
            start = random.randint(0, len(text_in_words) - canary_length)
            canary = " ".join(text_in_words[start:start + canary_length])
            canaries.append(canary)
            i += 1
    return canaries

def download_punkt_if_not_exists():
    try:
        # Check if 'punkt' tokenizer models are already available
        find('tokenizers/punkt')
        print("Punkt tokenizer is already downloaded.")
    except LookupError:
        # If not found, download the 'punkt' tokenizer models
        print("Punkt tokenizer not found. Downloading now...")
        nltk.download('punkt')
        print("Punkt tokenizer downloaded.")

def is_duplicate(seq: str, canaries: list, threshold: float = 0.2):
    seq_set = set(nltk.word_tokenize(seq.lower()))
    for canary in canaries:
        canary_set = set(nltk.word_tokenize(canary.lower()))
        if nltk.jaccard_distance(seq_set, canary_set) < threshold:
            return True
    return False

def make_canaries_label_compatible(canaries: list, original_dataset: datasets.Dataset, 
                             label_comptability_method: str, 
                             text_name: str, label_name: str):
    all_label_names = original_dataset.features[label_name].names
    all_label_ids = range(len(all_label_names))
    if label_comptability_method == 'uniform':
        # sample labels from the same distribution as the original dataset
        all_labels = [x[label_name] for x in original_dataset]
        label_counts = Counter(all_labels)
        total_count = len(all_labels)
        label_distribution = {label: count / total_count for label, count in label_counts.items()}

        canary_label_counts = {label: int(distribution * len(canaries)) for label, distribution in label_distribution.items()}
        # Ensure the canary label size matches exactly by adjusting for rounding errors
        adjustment = len(canaries) - sum(canary_label_counts.values())
        if adjustment != 0:
            # Adjust the largest label group by the difference
            largest_label = max(canary_label_counts, key=canary_label_counts.get)
            canary_label_counts[largest_label] += adjustment
        canary_labels = sum([[label] * count for label, count in canary_label_counts.items()], [])

        #now convert this to a dataset
        canary_dataset = datasets.Dataset.from_dict({
            text_name: canaries,
            label_name: canary_labels
        })
        canary_dataset = canary_dataset.cast_column(label_name, original_dataset.features[label_name])
        return canary_dataset, original_dataset

    elif label_comptability_method == 'extend':
        canary_labels = [max(all_label_ids) + 1] * len(canaries)
        canary_dataset = datasets.Dataset.from_dict({
            text_name: canaries,
            label_name: canary_labels
        })
        # now make the datasets compatible
        original_class_names = original_dataset.features[label_name]
        new_class_names = original_class_names.names + ["canary"] # by default set this label to canary
        new_label_feature = datasets.ClassLabel(names=new_class_names)

        # now cast both the canary and original dataset
        canary_dataset = canary_dataset.cast_column(label_name, new_label_feature)
        original_dataset_copy = datasets.Dataset.from_dict({
            text_name: original_dataset[text_name],
            label_name: original_dataset[label_name]
        })
        original_dataset_copy = original_dataset_copy.cast_column(label_name, new_label_feature)

        # check if all labels are now correct
        label_str2int = {label: id for id, label in enumerate(original_dataset_copy.features[label_name].names)}
        assert len(label_str2int) == len(original_dataset.features[label_name].names) + 1
        print("New label mapping: ", label_str2int)
        return canary_dataset, original_dataset_copy

    else:
        raise ValueError(f'Unknown label_comptability_method: {label_comptability_method}')
    
## all code for canaries with varying perplexity

# this function is a copy-paste from the tokenize_component in the main folder of dp-transformers
def main_preprocess_function(examples, tokenizer, sequence_len=256):
    batch_size = len(examples["prompt"])
    model_inputs = tokenizer(examples["prompt"])
    labels = tokenizer(examples["completion"])
        
    # Concatenate the prompt and completion parts as one input and set -100 to the labels of the prompt part
    # This is because only the completion part will be used to calculate the loss
    for i in range(batch_size):
        sample_input_ids = model_inputs["input_ids"][i]
        # Tokenizer adds <s> to input_ids so just take the rest
        label_input_ids = labels["input_ids"][i][1:]
        model_inputs["input_ids"][i] = sample_input_ids + label_input_ids + [tokenizer.eos_token_id]
        labels["input_ids"][i] = [-100] * len(sample_input_ids) + label_input_ids + [tokenizer.eos_token_id]
        model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])

    # Pad the samples with sequence_len and trim if longer than sequence_len
    # NOTE THAT IF CONTEXT IS LONGER THAN SEQUENCE_LEN, THERE WILL BE NOTHING TO PREDICT, LABEL IS ALL -100
    for i in range(batch_size):
        sample_input_ids = model_inputs["input_ids"][i]
        label_input_ids = labels["input_ids"][i]
        model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (sequence_len - len(sample_input_ids)) \
                                        + sample_input_ids
        model_inputs["attention_mask"][i] = [0] * (sequence_len - len(sample_input_ids)) \
                                            + model_inputs["attention_mask"][i]
        labels["input_ids"][i] = [-100] * (sequence_len - len(sample_input_ids)) + label_input_ids
        model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:sequence_len])
        model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:sequence_len])
        labels["input_ids"][i] = torch.tensor(labels["input_ids"][i][:sequence_len])

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def compute_perplexity_batch(
    model: AutoModelForCausalLM,
    batch: Dict[str, torch.Tensor],
    device:torch.device):
    '''
    We compute the perplexity of a batch of sequences.
    Specifically we set the attention-mask equal to 1 for the prompt, extract the model logits for the completion
    and compute the perplexity of only the completion.
    '''

    with torch.no_grad():
        
        input_ids = torch.tensor(batch["input_ids"]).to(device)
        attention_mask = torch.tensor(batch["attention_mask"]).to(device)
        labels =  torch.tensor(batch["labels"])

        output = model(input_ids, attention_mask=attention_mask)

        logits = output.logits[:, :-1, :] # remove the last token prediction
        labels = labels[:, 1:] # Remove the first token in the labels
        labels_np = labels.cpu().numpy()        

        # pytorch expects n_classes dimension as the 2nd (i.e. index 1) dimension
        logits = np.swapaxes(logits.cpu().numpy(), 1, -1)

        loss = CrossEntropyLoss(ignore_index=-100, reduction="none")(torch.tensor(logits), torch.tensor(labels)).cpu().numpy()
    
    completion_mask = attention_mask.cpu().numpy()[:, 1:]
    completion_mask[labels_np == -100] = 0
    mean_cross_entropy_loss = loss.mean(axis=1, where=completion_mask.astype(bool))
    ppls = np.exp(mean_cross_entropy_loss)

    return ppls

def get_random_canaries(tokenizer: AutoTokenizer, n_canaries: int, canary_length: int,
                         n_tokens = 200):
    vocab = [k for k in tokenizer.get_vocab().values()]
    canaries = []
    for i in range(n_canaries):
        random_canary_tokens = []
        for j in range(n_tokens):
            token = random.choice(vocab)
            random_canary_tokens.append(token)
        random_canary_text = tokenizer.decode(random_canary_tokens)
        random_canary_truncated = " ".join(random_canary_text.split()[:canary_length])
        canaries.append(random_canary_truncated)
    return canaries

def update_temperature(all_ppls, min_temperature, max_temperature, min_ppl, max_ppl, learning_rate=0.1, shrink_factor=0.9):

    # Calculate the mean perplexity of the current batch
    mean_ppl = np.mean(all_ppls)

    # Compute the difference from the target perplexity range
    lower_diff = min_ppl - mean_ppl
    upper_diff = mean_ppl - max_ppl

    # Adjust temperatures based on the mean perplexity
    if mean_ppl < min_ppl:
        # If perplexity is below the target, increase the temperature
        adjustment = learning_rate * abs(lower_diff / min_ppl)
        min_temperature = min(min_temperature + adjustment, max_temperature)
        max_temperature = min(max_temperature + adjustment, 2 * max_temperature)  # prevent excessive growth
    elif mean_ppl > max_ppl:
        # If perplexity is above the target, decrease the temperature
        adjustment = learning_rate * abs(upper_diff / max_ppl)
        max_temperature = max(max_temperature - adjustment, min_temperature)
        min_temperature = max(min_temperature - adjustment, 0.5 * min_temperature)  # prevent negative temps
    else:
        # If within the target range, consider narrowing the range
        range_center = (min_temperature + max_temperature) / 2
        range_width = (max_temperature - min_temperature) * shrink_factor
        min_temperature = range_center - range_width / 2
        max_temperature = range_center + range_width / 2

    # Print the new temperature range
    print(f"New temperature range: {min_temperature:.4f} - {max_temperature:.4f}")

    return min_temperature, max_temperature

def generate_synthetic_canaries_ppl(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, 
                                n_canaries: int, canary_length: int,
                                prompt: str, min_ppl: float, max_ppl: float,
                                min_temperature: float, max_temperature: float,
                                batch_size: int, device: torch.device):

    if max_ppl == -1:
        print("You have provided max_ppl=-1, so generating canaries with random tokens")
        canaries = get_random_canaries(tokenizer, n_canaries, canary_length)
        return canaries

    canaries = [] 
    inputs = tokenizer([prompt] * batch_size, return_tensors="pt").to(device)

    total_samples = 0
    step = 0
    duplicates = 0

    while len(canaries) < n_canaries:
        if step > 0 and step % 5 == 0:
            samples = len(canaries)
            print(
                f"Step: {step} | total: {total_samples} | accepted: {samples} | duplicates: {duplicates}"
            )
            if len(canaries) > 0:
                print('The last canary was: ')
                print(canaries[-1])

        # sample one temperature for the entire batch
        temperature = (min_temperature + max_temperature) / 2.0

        generated_ids = model.generate(
            **inputs,
            max_length=canary_length * 2, # we define canary length in words, so we need to generate a bit more
            do_sample=True,
            temperature=temperature,
            top_p=1.0,
            top_k=0,
            pad_token_id=tokenizer.eos_token_id 
        )

        # now we have to get the generated text 
        generated_text = tokenizer.batch_decode(generated_ids[:, 1:])
        valid_text = []
        for text in generated_text:
            # remove prompt and everything from eos token
            text = text.replace(prompt, '')
            text = text.split(tokenizer.eos_token)[0]
            text_split = text.split()
            n_words = len(text_split)
            if is_duplicate(text, canaries):
                duplicates += 1
                continue
            if n_words >= canary_length:
                valid_text.append(" ".join(text_split[:canary_length]))

        if min_ppl == max_ppl:
            # if we are not controlling perplexity, then we can just add all the generated text
            n_canaries_needed = n_canaries - len(canaries)
            canaries.extend(valid_text[:n_canaries_needed])
        
        else:
            # now we have to compute the perplexity of these selected texts
            valid_text_dataset = datasets.Dataset.from_dict({'prompt':[prompt] * len(valid_text), 'completion': valid_text})
            tokenized_valid_text = valid_text_dataset.map(lambda x: main_preprocess_function(x, tokenizer=tokenizer), batched=True, 
                                                        num_proc=10, desc="tokenizing dataset", 
                                                        remove_columns=valid_text_dataset.column_names)
            all_ppls = compute_perplexity_batch(model, tokenized_valid_text, device)
            
            for idx, text in enumerate(valid_text):
                ppl = all_ppls[idx]
                print(temperature, ppl, text)
                if ppl >= min_ppl and ppl <= max_ppl and len(canaries) < n_canaries:
                    canaries.append(text)

            # optimize temperature for next batch
            min_temperature, max_temperature =  update_temperature(all_ppls, min_temperature, max_temperature, min_ppl, max_ppl)

        print(f"Found {len(canaries)} canaries - continuing...")
        total_samples += batch_size
        step += 1

    return canaries

def get_ppl_controlled_canaries(original_dataset: datasets.Dataset, label_comptability_method: str, 
                                text_name: str, label_name: str,
                                model: AutoModelForCausalLM, tokenizer: AutoTokenizer, 
                                n_canaries: int, canary_length: int,
                                templated_prompt: str, min_ppl: float, max_ppl: float,
                                min_temperature: float, max_temperature: float,
                                batch_size: int, device: torch.device):
    
    all_label_names = original_dataset.features[label_name].names
    all_label_ids = range(len(all_label_names))
    if label_comptability_method == 'uniform':
        # sample labels from the same distribution as the original dataset
        all_labels = [x[label_name] for x in original_dataset]
        label_counts = Counter(all_labels)
        total_count = len(all_labels)
        label_distribution = {label: count / total_count for label, count in label_counts.items()}

        canary_label_counts = {label: int(distribution * n_canaries) for label, distribution in label_distribution.items()}
        # Ensure the canary label size matches exactly by adjusting for rounding errors
        adjustment = n_canaries - sum(canary_label_counts.values())
        if adjustment != 0:
            # Adjust the largest label group by the difference
            largest_label = max(canary_label_counts, key=canary_label_counts.get)
            canary_label_counts[largest_label] += adjustment
        
        canaries = []
        for label, count in canary_label_counts.items():
            label_str = all_label_names[label]
            adapted_prompt = templated_prompt.replace(f"{{{label_name}}}", label_str)
            # generate canaries for this label, controlled by perplexity
            # resulting canaries is a list of textual canaries (without the prompt)
            canaries_label = generate_synthetic_canaries_ppl(model=model, tokenizer=tokenizer, 
                                n_canaries=count, canary_length=canary_length, prompt=adapted_prompt,
                                min_ppl=min_ppl, max_ppl=max_ppl, min_temperature=min_temperature, max_temperature=max_temperature,
                                batch_size=batch_size, device=device)
            canaries.extend(canaries_label)
        
        canary_labels = sum([[label] * count for label, count in canary_label_counts.items()], [])

        # now convert this to a dataset
        canary_dataset = datasets.Dataset.from_dict({
            text_name: canaries,
            label_name: canary_labels
        })
        canary_dataset = canary_dataset.cast_column(label_name, original_dataset.features[label_name])
        return canary_dataset, original_dataset

    elif label_comptability_method == 'extend':
        canary_labels = [max(all_label_ids) + 1] * n_canaries

        # now generate canaries with this label
        adapted_prompt = templated_prompt.replace(f"{{{label_name}}}", "canary")
        canaries = generate_synthetic_canaries_ppl(model=model, tokenizer=tokenizer, 
                                n_canaries=n_canaries, canary_length=canary_length, prompt=adapted_prompt,
                                min_ppl=min_ppl, max_ppl=max_ppl, min_temperature=min_temperature, max_temperature=max_temperature,
                                batch_size=batch_size, device=device)

        canary_dataset = datasets.Dataset.from_dict({
            text_name: canaries,
            label_name: canary_labels
        })

        # now make the datasets compatible
        original_class_names = original_dataset.features[label_name]
        new_class_names = original_class_names.names + ["canary"] # by default set this label to canary
        new_label_feature = datasets.ClassLabel(names=new_class_names)

        # now cast both the canary and original dataset
        canary_dataset = canary_dataset.cast_column(label_name, new_label_feature)
        original_dataset_copy = datasets.Dataset.from_dict({
            text_name: original_dataset[text_name],
            label_name: original_dataset[label_name]
        })
        original_dataset_copy = original_dataset_copy.cast_column(label_name, new_label_feature)

        # check if all labels are now correct
        label_str2int = {label: id for id, label in enumerate(original_dataset_copy.features[label_name].names)}
        assert len(label_str2int) == len(original_dataset.features[label_name].names) + 1
        print("New label mapping: ", label_str2int)
        return canary_dataset, original_dataset_copy

    else:
        raise ValueError(f'Unknown label_comptability_method: {label_comptability_method}')
    