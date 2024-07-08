import datasets 
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
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

def generate_synthetic_canaries(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, 
                                n_canaries: int, canary_length: int,
                                temperature: float, 
                                batch_size: int, device: torch.device):

    canaries = [] 
    input = tokenizer([""] * batch_size, return_tensors="pt").to(device)

    total_samples = 0
    step = 0
    duplicates = 0

    while len(canaries) < n_canaries:
        if step > 0 and step % 5 == 0:
            samples = len(canaries)
            logging.info(
                f"Step: {step} | total: {total_samples} | accepted: {samples} | duplicates: {duplicates}"
            )
            if len(canaries) > 0:
                print('The last canary was: ')
                print(canaries[-1])

        generated_ids = model.generate(
            input["input_ids"],
            max_length=canary_length * 2, # we define canary length in words, so we need to generate a bit more
            do_sample=True,
            temperature=temperature,
        )

        generated_text = tokenizer.batch_decode(generated_ids[:, 1:])

        for text in generated_text:
            # only consider text before any eos token
            text = text.split(tokenizer.eos_token)[0]
            text_split = text.split()
            n_words = len(text_split)
            if n_words < canary_length:
                continue
            else:
                # now sample a random substring of length canary_length
                text_in_words = text.split()
                start = random.randint(0, len(text_in_words) - canary_length)
                canary = " ".join(text_in_words[start:start + canary_length])

                if is_duplicate(canary, canaries):
                    duplicates += 1
                    continue
                canaries.append(canary)

        step += 1

    return canaries

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
    