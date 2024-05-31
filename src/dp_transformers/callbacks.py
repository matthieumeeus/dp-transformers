import logging

from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState
from transformers.training_args import TrainingArguments
from transformers.utils.logging import get_logger, log_levels
from datetime import datetime
from accelerate import PartialState
from typing import Dict, Any, Literal


class DistributedTrainLoggerCallback(TrainerCallback):
    def __init__(self, log_level: str, log_on_root: bool = True, formatter: logging.Formatter = None):
        if log_level not in log_levels.keys():
            raise ValueError(f"Invalid log level {log_level}. Valid log levels are {log_levels.keys()}")
        self.log_on_root = log_on_root
        self.logger = get_logger("TrainLogger")

        handler = logging.StreamHandler()
        handler.setLevel(log_levels[log_level])

        process_index = PartialState().process_index

        if formatter is None:
            formatter = logging.Formatter(f"[%(levelname)s | Proc {process_index}] %(asctime)s >> %(message)s")
        handler.setFormatter(formatter)

        self.logger.addHandler(handler)
        self.logger.setLevel(log_levels[log_level])
        
    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.log(args, "Starting training")

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.log(args, "Training completed")

    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.log(args, f"Starting epoch {state.epoch}")

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.log(args, f"Epoch {state.epoch} completed")

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs: Dict[str, Any]):
        if self.log_on_root and args.process_index == 0:
            self.log(args, str(logs))

    def log(self, args: TrainingArguments, message: str):
        self.logger.info(message)
