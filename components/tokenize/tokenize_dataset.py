import os
import sys
import datasets
import transformers
import logging

from dataclasses import dataclass, field
from pathlib import Path
from typing import Union, Optional
from multiprocessing import cpu_count

from data_utils import MyDataset, MyChatDataset
from constant_length_dataset import ConstantLengthDataset

logger = logging.getLogger(__name__)


@dataclass
class Arguments:
    tokenizer_name_or_path: Union[str, Path] = field(metadata={
        "help": "Tokenizer name in HuggingFace, e.g. 'gpt2'"
    })
    data_path: Path = field(metadata={
        "help": "Path to training data in jsonl format"
    })
    tokenized_data_path: Path = field(metadata={
        "help": "Path to tokenized data"
    })
    packed_dataset: bool = field(default=False, metadata={
        "help": "Whether the dataset will be packed or not"
    })
    chat_format: bool = field(default=False, metadata={
        "help": "Whether the dataset should be processed chat format or not"
    })
    sequence_len: int = field(default=128, metadata={
        "help": "Maximum sequence length"
    })
    nproc: Optional[int] = field(default=None, metadata={
        "help": "Number of processes to use for tokenization"
    })
    trust_remote_code: bool = field(default=False, metadata={
        "help": "Trust remote code"
    })
    cache_data_path: Path = field(default=Path("/tmp/cache/datasets"), metadata={
        "help": "Path to cache data"
    })


def main(args: Arguments) -> int:
    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer_name_or_path, 
                                                           trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if args.packed_dataset:
        logger.info(f"Loading a dataset and packing the data into a constant length dataset.")
        train_data_path = str(args.data_path)
        if os.path.isdir(train_data_path):
            files = [os.path.join(train_data_path, f) for f in os.listdir(train_data_path)]
        else:
            files = [train_data_path]

        dataset = datasets.DatasetDict({
            "train": datasets.Dataset.from_json(files, cache_dir=args.cache_data_path),
        })

        constant_length_iterator = ConstantLengthDataset(
                tokenizer,
                dataset["train"],
                dataset_text_field='completion',
                formatting_func=None,
                seq_length=args.sequence_len,
                infinite=False,
                eos_token_id=tokenizer.eos_token_id,
            )

        def data_generator(constant_length_iterator):
            yield from constant_length_iterator

        try:
            dataset = datasets.Dataset.from_generator(
                data_generator, gen_kwargs={"constant_length_iterator": constant_length_iterator},
                cache_dir=args.cache_data_path,
            )
        except:
            raise ValueError(
                "Error occurred while packing the dataset. "
                "Make sure that your dataset has enough samples to at least yield one packed sequence."
            )
        dataset.save_to_disk(args.tokenized_data_path)
    else:
        # Load dataset
        if args.chat_format:
            logger.info(f"Loading a dataset in a chat format (e.g. system-user-assistant)")
            dataset = MyChatDataset(args.data_path, tokenizer, args.sequence_len)
        else:
            logger.info(f"Loading a dataset in a plain format (i.e. prompt-completion)")
            dataset = MyDataset(args.data_path, tokenizer, args.sequence_len, cache_data_path=str(args.cache_data_path))

        # Tokenize data
        num_proc = cpu_count()
        dataset.dataset = dataset.dataset.map(
        dataset.preprocess_function, batched=True, num_proc=num_proc, desc="tokenizing dataset", 
            remove_columns=dataset.dataset.column_names['train']
        )
        dataset.dataset["train"].save_to_disk(args.tokenized_data_path)

    return 0


if __name__ == "__main__":
    parser = transformers.HfArgumentParser((Arguments,))
    args, = parser.parse_args_into_dataclasses()
    sys.exit(main(args))
