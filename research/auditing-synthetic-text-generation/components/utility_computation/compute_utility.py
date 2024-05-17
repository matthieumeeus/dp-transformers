
'''Trained roberta sentiment classifier on the synthetic data to compute utility'''

from pathlib import Path
import transformers
import datasets
import dp_transformers
import sys
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

def prep_data(args, dataset, text_name, label_name, tokenizer, label_to_id = None):
    
    original_col_names = dataset.column_names
    
    # Tokenize data
    def tokenize_function(examples):
        return tokenizer(examples[text_name], padding="max_length", truncation=True)

    tokenized_data = dataset.map(
            tokenize_function, batched=True, num_proc=8, desc="tokenizing dataset",
        )

    # Add labels to the tokenized dataset
    if label_to_id is None:
        class_label = datasets.ClassLabel(names=list(set(dataset[label_name])))
        label_to_id = {label: id for id, label in enumerate(class_label.names)}

    def add_labels(example):
        example['label'] = label_to_id[example[label_name]]
        return example
    
    tokenized_data = tokenized_data.map(add_labels, batched=False,
                                        remove_columns=[col for col in original_col_names if col != 'label'])
    tokenized_data.set_format('torch')
    return tokenized_data, label_to_id

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

    # Load dataset for now assuming it's only one csv file of generated data
    train_data = datasets.load_from_disk(str(args.data.train_data_path), keep_in_memory=True)
    eval_data = datasets.load_from_disk(str(args.data.eval_data_path), keep_in_memory=True)

    tokenized_train_data, label_to_id = prep_data(args, train_data, args.data.train_text_name, 
                                                  args.data.train_label_name, tokenizer)
    tokenized_eval_data, _ = prep_data(args, eval_data, args.data.eval_text_name, 
                                       args.data.eval_label_name, tokenizer, label_to_id)

    # Load the model
    model = transformers.RobertaForSequenceClassification.from_pretrained(args.model.model_name_or_path, 
                                                                           num_labels=len(label_to_id))
    
    # Define training arguments
    trainer = transformers.Trainer(
        args=args.train,
        model=model,
        train_dataset=tokenized_train_data,
        eval_dataset=tokenized_eval_data,
        compute_metrics=compute_metrics,
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

