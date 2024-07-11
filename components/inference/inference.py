# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

'''Train LLMs without DP using QLoRA'''
import datasets
import transformers
import sys
import logging
import torch
import numpy as np
import multiprocess as mp
import copy

from torch import nn
from peft import PeftModel
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from pathlib import Path
from ast import literal_eval
from enum import Enum
from privacy_estimates.experiments.attacks.signals import Signal, SIGNALS


logger = logging.getLogger(__name__)
mp.set_start_method('spawn', force=True)  # force=True can be used to reset the method if needed elsewhere

TORCH_DTYPES = {
    "fp16": torch.float16,
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
}


class AggregationMethod(Enum):
    MEAN = "mean"
    MAX = "max"
    MIN = "min"
    SUM = "sum"
    SUMLOG = "sumlog"
    EXPSUM = "expsum"


@dataclass
class Arguments:
    base_model_path: Optional[Path] = field(default=None, metadata={
        "help": "Path to the base model"
    })
    peft_path: Optional[Path] = field(default=None, metadata={
        "help": "Path to the PEFT model"
    })
    per_device_batch_size: int = field(default=8, metadata={
        "help": "Batch size per device"
    })
    torch_dtype: str = field(default="bf16", metadata={
        "help": "Data type for model"
    })
    trust_remote_code: bool = field(default=False, metadata={
        "help": "Whether to trust remote code when loading model from HuggingFace."
    })
    tokenized_data_path: Optional[Path] = field(default=None, metadata={
        "help": "Path to tokenized data in HF dataset format"
    })
    predictions_path: Optional[Path] = field(default=None, metadata={
        "help": "Path to save predictions"
    })
    log_level: str = field(default="INFO", metadata={
        "help": "Log level"
    })
    use_cpu: bool = field(default=False, metadata={
        "help": "Whether to use CPU for training"
    })
    mi_signal_method: Optional[str] = field(default=None, metadata={
        "help": "Method to compute MI signal", "choices": list(SIGNALS.keys())
    })
    mi_signal_extra_args: Optional[str] = field(default=None, metadata={
        "help": "Extra arguments for MI signal method"
    })
    mi_signal_aggregation: Optional[str] = field(default=None, metadata={
        "help": "Method to aggregate MI signal", "choices": [a.value for a in AggregationMethod]
    })
    disable_distributed: bool = field(default=False, metadata={
        "help": "Whether to disable distributed inference."
    })

    def __post_init__(self):
        self.log_level = logging.getLevelName(self.log_level.upper())
        if self.mi_signal_extra_args is not None:
            self.mi_signal_extra_args = {
                a.split("=")[0]: literal_eval(a.split("=")[1]) for a in self.mi_signal_extra_args.split()
            }
        if self.torch_dtype is not None and isinstance(self.torch_dtype, str):
            if self.torch_dtype not in TORCH_DTYPES:
                 raise ValueError(f"Invalid torch dtype: {self.torch_dtype}. Must be one of {list(TORCH_DTYPES.keys())}")
            self.torch_dtype = TORCH_DTYPES[self.torch_dtype]

def aggregate_mi_signal(mi_signal: np.ndarray, completion_mask: np.ndarray, aggregation_method: str) -> np.ndarray:
    """
    Apply aggregation over the sequence length.

    Note: It is important to only apply functions that preserve monotonicity since the convention that larger
    values of the mi_signal are evidence for in-membership should be maintained.
    """
    completion_mask = completion_mask.astype(bool)
    assert mi_signal.ndim == 2
    aggregation_method = AggregationMethod(aggregation_method)
    match aggregation_method:
        case AggregationMethod.MEAN:
            return mi_signal.mean(axis=1, where=completion_mask)
        case AggregationMethod.MAX:
            return mi_signal.max(axis=1, where=completion_mask)
        case AggregationMethod.MIN:
            return mi_signal.min(axis=1, where=completion_mask)
        case AggregationMethod.SUM:
            return mi_signal.sum(axis=1, where=completion_mask)
        case AggregationMethod.SUMLOG:
            return np.log(mi_signal, where=completion_mask).sum(axis=1, where=completion_mask)
        case AggregationMethod.EXPSUM:
            return np.exp(mi_signal.astype(np.longdouble).sum(axis=1, where=completion_mask))
        case _:
            raise ValueError(f"Invalid aggregation method: {aggregation_method}")

class DistributedEvaluator:
    def __init__(self, model: nn.Module, devices: List[str], signal_method: Signal, signal_aggregation: str):
        logger.info(f"Initializing evaluator on devices {devices}")
        self.models = [copy.deepcopy(model).to(d) for d in devices]
        self.devices = devices
        self.signal_method = signal_method
        self.signal_aggregation = signal_aggregation

    def evaluate(self, batch: Dict[str, torch.Tensor], rank: Optional[int] = None) -> Dict[str, torch.Tensor]:
        if rank is None:
            rank = 0
        device = self.devices[rank]
        model = self.models[rank]
        model.eval()
        
        with torch.no_grad():
            batch = {k: v.to(device) for k, v in batch.items()}
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]

            output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            attention_mask_np = attention_mask.cpu().numpy()

        # manual patch - converting all attention masks to 0 where labels is -100
        labels_np = labels.cpu().numpy()
        attention_mask_np[labels_np == -100] = 0

        mi_signal_seq = self.signal_method.compute_mi_signal_from_logits(
            logits=output.logits.cpu().numpy(), labels=labels.cpu().numpy(), attention_mask=attention_mask_np
        )
        mi_signal = aggregate_mi_signal(
            mi_signal_seq, completion_mask=attention_mask_np, aggregation_method=self.signal_aggregation
        )
        assert np.isnan(mi_signal).any() == False, "NaN values in MI signal"
        return {"mi_signal": mi_signal.astype(np.double), "log_mi_signal": np.log(mi_signal).astype(np.double)}


def main(args: Arguments):
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = args.log_level
    logging.getLogger().setLevel(level=log_level)
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info(f"Parameters: {args}")
    logger.info(f"MP start method: {mp.get_start_method()}")

    # Load dataset
    dataset: datasets.Dataset = datasets.load_from_disk(args.tokenized_data_path, keep_in_memory=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # Load model
    logger.info(f"Loading model: {args.base_model_path}")

    model = transformers.AutoModelForCausalLM.from_pretrained(
        str(args.base_model_path), trust_remote_code=args.trust_remote_code, torch_dtype=args.torch_dtype
    )
    if args.peft_path is not None:
        logger.info(f"Loading PEFT model: {args.peft_path}")
        model = PeftModel.from_pretrained(model, args.peft_path)

    # Set-up MI signal method
    mi_signal_method = SIGNALS[args.mi_signal_method](**args.mi_signal_extra_args)

    if args.use_cpu:
        devices = [torch.device("cpu")]
    else:
        assert torch.cuda.is_available()
        devices = [torch.device("cuda", i) for i in range(torch.cuda.device_count())]

    num_proc = len(devices)
    if args.disable_distributed:
        num_proc = None

    evaluator = DistributedEvaluator(model=model, devices=devices, signal_method=mi_signal_method,
                                     signal_aggregation=args.mi_signal_aggregation)
    print('Batch size: ', args.per_device_batch_size)
    print('Dataset: ', dataset[0])
    results = dataset.map(
        evaluator.evaluate,
        batched=True, batch_size=args.per_device_batch_size,
        num_proc=num_proc, remove_columns=dataset.column_names,
        with_rank=True
    )
    results.set_format(None)
    assert len(results) == len(dataset)
    results.save_to_disk(args.predictions_path)


if __name__ == "__main__":
    arg_parser = transformers.HfArgumentParser((Arguments,))
    args, = arg_parser.parse_args_into_dataclasses()
    main(args)
