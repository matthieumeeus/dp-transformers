import os
import datasets
import torch
import copy

from pathlib import Path
from typing import Union


# Modified from https://huggingface.co/docs/peft/task_guides/clm-prompt-tuning
def main_preprocess_function(examples, tokenizer, sequence_len):
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


class MyDataset:
    """ This class is used to load and preprocess the dataset.
        We assume the dataset is in json format with the following fields:
        - prompt: the text to be prompted to the model
        - completion: the completion to be included in the loss function
    """

    def __init__(self, train_data_path, tokenizer, sequence_len, cache_data_path):
        # Load data
        train_data_path = str(train_data_path)
        if os.path.isdir(train_data_path):
            files = [os.path.join(train_data_path, f) for f in os.listdir(train_data_path)]
        else:
            files = [train_data_path]

        self.dataset = datasets.DatasetDict({
            "train": datasets.Dataset.from_json(files, cache_dir=cache_data_path),
        })
        self.tokenizer = tokenizer
        self.sequence_len = sequence_len

    def preprocess_function(self, example):
        return main_preprocess_function(example, self.tokenizer, self.sequence_len)


# Modified from https://huggingface.co/docs/peft/task_guides/clm-prompt-tuning
def main_preprocess_function_chat(examples, tokenizer, sequence_len):
    batch_size = len(examples["chat"])
    
    # Chat is already tokenized
    # For now we take into both prompt and completion in the loss function
    model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
    model_inputs["input_ids"] = copy.deepcopy(examples["chat"])
    model_inputs["attention_mask"] = [[1] * len(chat) for chat in examples["chat"]]
    model_inputs["labels"] = copy.deepcopy(examples["chat"])

    # Pad the samples with sequence_len and trim if longer than sequence_len
    # NOTE THAT IF CONTEXT IS LONGER THAN SEQUENCE_LEN, THERE WILL BE NOTHING TO PREDICT, LABEL IS ALL -100
    for i in range(batch_size):
        sample_input_ids = model_inputs["input_ids"][i]
        label_input_ids = model_inputs["labels"][i]
        model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (sequence_len - len(sample_input_ids)) \
                                        + sample_input_ids
        model_inputs["attention_mask"][i] = [0] * (sequence_len - len(sample_input_ids)) \
                                            + model_inputs["attention_mask"][i]
        model_inputs["labels"][i] = [-100] * (sequence_len - len(sample_input_ids)) + label_input_ids
        model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:sequence_len])
        model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:sequence_len])
        model_inputs["labels"][i] = torch.tensor(model_inputs["labels"][i][:sequence_len])

    return model_inputs


class MyChatDataset:
    """ This class is used to load and preprocess the dataset.
        We assume the dataset is in json format with the following fields:
        - prompt: the text to be prompted to the model
        - completion: the completion to be included in the loss function
        We also assume that the dataset will be used to fine-tune a chat model
        https://huggingface.co/docs/transformers/chat_templating
    """

    def __init__(self, train_data_path, tokenizer, sequence_len):
        # Load data
        train_data_path = str(train_data_path)
        if os.path.isdir(train_data_path):
            files = [os.path.join(train_data_path, f) for f in os.listdir(train_data_path)]
        else:
            files = [train_data_path]

        self.dataset = datasets.DatasetDict({
            "train": datasets.Dataset.from_json(files),
        })
        self.tokenizer = tokenizer
        self.sequence_len = sequence_len
        self.dataset = self.dataset.map(
            lambda x: {"chat": 
                       [self.tokenizer.apply_chat_template([{"role": "system", "content": ""},
                                                            {"role": "user", "content": q}, 
                                                            {"role": "assistant", "content": s}], tokenize=True) 
                                                            for q, s in zip(x["prompt"], x["completion"])]},
            batched=True,
            num_proc=None,
        )

    def preprocess_function(self, example):
        return main_preprocess_function_chat(example, self.tokenizer, self.sequence_len)
