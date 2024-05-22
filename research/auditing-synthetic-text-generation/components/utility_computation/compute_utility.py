
'''Trained roberta sentiment classifier on the synthetic data to compute utility'''

from pathlib import Path
import transformers
import datasets
import dp_transformers
import sys
import os
import logging
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import numpy as np

from dataclasses import dataclass, field
from typing import Optional, Union

logger = logging.getLogger(__name__)

@dataclass
class RobertaModelArguments:
    model_name_or_path: Union[str, Path] = field(default="roberta-base", metadata={
        "help": "Model name in HuggingFace, e.g. 'roberta-base'"
    })
    sequence_len: int = field(default=128, metadata={
        "help": "Maximum sequence length"
    })

@dataclass
class DataArguments:
    is_synthetic:  bool = field(metadata={
        "help": "Whether the training data is synthetic or not"
    })
    templated_prompt: str = field(default="This is a {{label}} sentence", metadata={
        "help": "Prompt with a placeholder for the label"
    })
    train_data_path:  Optional[Path] = field(default=None, metadata={
        "help": "Path to training data in csv format"
    })
    train_label_name: str = field(default="Prompt", metadata={
        "help": "Name of the label column in the dataset"
    })
    train_text_name: str = field(default="Generation", metadata={
        "help": "Name of the text column in the dataset"
    })
    eval_data_path:  Optional[Path] = field(default=None, metadata={
        "help": "Path to training data in csv format"
    })
    eval_label_name: str = field(default="text", metadata={
        "help": "Name of the label column in the dataset"
    })
    eval_text_name: str = field(default="label", metadata={
        "help": "Name of the text column in the dataset"
    })

@dataclass
class Arguments:
    train: dp_transformers.TrainingArguments
    model: RobertaModelArguments
    data: DataArguments

def load_synthetic_data(data_path: str,
                        og_label_name: str, new_label_name: str,
                        og_text_name: str, new_text_name: str,
                        label_str2int: dict, templated_prompt: str):
    path_to_csvs = [file for file in os.listdir(data_path) if file.endswith('.csv')]
    all_datasets = []
    for path in path_to_csvs:
        dataset = datasets.load_dataset('csv', data_files={'train':os.path.join(data_path, path)})
        all_datasets.append(dataset['train'])
    full_dataset = datasets.concatenate_datasets(all_datasets)

    # now make the labels compatible with the eval dataset
    prompt_to_label = {}
    for prompt in set(full_dataset[og_label_name]):
        for label_str in label_str2int.keys():
            if label_str in prompt:
                prompt_to_label[prompt] = label_str2int[label_str]
                break
    if len(prompt_to_label) != len(set(full_dataset[og_label_name])):
        raise ValueError('Not all labels found in the label mapping')
    
    full_dataset = full_dataset.map(lambda x: {new_text_name: x[og_text_name], new_label_name: prompt_to_label[x[og_label_name]]}, 
                                    remove_columns=[og_text_name, og_label_name])

    return full_dataset

def prep_data(args, dataset, text_name, label_name, tokenizer, return_mapping = False):
    
    original_col_names = dataset.column_names
    
    # Tokenize data
    def tokenize_function(examples):
        return tokenizer(examples[text_name], padding="max_length", truncation=True)

    tokenized_data = dataset.map(
            tokenize_function, batched=True, num_proc=8, desc="tokenizing dataset",
        )

    # Get the mapping if needed
    if return_mapping:
        # in this case, get the label mapping from the dataset (eval dataset)
        class_labels = dataset.features[label_name]
        label_str2int = {label: id for id, label in enumerate(class_labels.names)}
        print(label_str2int)
    else:
        label_str2int = None

    tokenized_data = tokenized_data.remove_columns([col for col in original_col_names if col != 'label'])
    tokenized_data.set_format('torch')

    return tokenized_data, label_str2int

def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    labels = p.label_ids
    probs = p.predictions
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')

    # Calculate AUC
    if len(np.unique(labels)) == 2:
        auc = roc_auc_score(labels, probs[:, 1])
    else:
        auc = roc_auc_score(labels, probs, multi_class='ovr')

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
    }

def main(args: Arguments):

    transformers.set_seed(args.train.seed)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = args.train.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {args.train.local_rank}, device: {args.train.device}, n_gpu: {args.train.n_gpu}, "
        f"distributed training: {bool(args.train.local_rank != -1)}, 16-bits training: {args.train.fp16}"
    )
    logger.info(f"Training/evaluation parameters {args.train}")
    logger.info(f"Model parameters {args.model}")

    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model.model_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load datasets
    # start with the evaluation dataset - allowing us to get the right label mapping
    eval_data = datasets.load_from_disk(str(args.data.eval_data_path), keep_in_memory=True)
    tokenized_eval_data, label_str2int = prep_data(args, eval_data, args.data.eval_text_name, 
                                       args.data.eval_label_name, tokenizer, return_mapping=True)
    
    if args.data.is_synthetic:
        train_data = load_synthetic_data(data_path=str(args.data.train_data_path),
                                         og_label_name=args.data.train_label_name, new_label_name=args.data.eval_label_name,
                                         og_text_name=args.data.train_text_name, new_text_name=args.data.eval_text_name,
                                         label_str2int=label_str2int, templated_prompt=args.data.templated_prompt)
        for i in range(10):
            print(i, train_data[i])
            print('---')
        tokenized_train_data, _ = prep_data(args, train_data, args.data.eval_text_name, 
                                            args.data.eval_label_name, tokenizer)
    else:
        train_data = datasets.load_from_disk(str(args.data.train_data_path), keep_in_memory=True)
        for i in range(10):
            print(i, train_data[i])
            print('---')
        tokenized_train_data, _ = prep_data(args, train_data, args.data.train_text_name, 
                                            args.data.train_label_name, tokenizer)

    # Load the model
    model = transformers.RobertaForSequenceClassification.from_pretrained(args.model.model_name_or_path, 
                                                                           num_labels=len(label_str2int))
    
    # Define training arguments
    trainer = transformers.Trainer(
        args=args.train,
        model=model,
        train_dataset=tokenized_train_data,
        eval_dataset=tokenized_eval_data,
        #compute_metrics=compute_metrics,
    )

    # Train the model
    trainer.train()
    
    trainer.save_model()


if __name__ == "__main__":
    arg_parser = transformers.HfArgumentParser(
        (dp_transformers.TrainingArguments, RobertaModelArguments, DataArguments)
    )
    train_args, model_args, data_args = arg_parser.parse_args_into_dataclasses()
    main(Arguments(train=train_args, model=model_args, data=data_args))

