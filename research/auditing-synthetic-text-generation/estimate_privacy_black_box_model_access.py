from azure.ai.ml import Input, Output
from pathlib import Path
from dataclasses import dataclass, asdict

from privacy_estimates.experiments.loaders import InferenceComponentLoader, TrainingComponentLoader, AMLComponentLoader
from privacy_estimates.experiments.games.black_box_membership_inference import (
    BlackBoxMembershipInferenceGameBase, GameConfig, ShadowModelConfig
)
from privacy_estimates.experiments.games.configs import MISignalConfig
from privacy_estimates.experiments.attacks.rmia import RmiaLoader, RmiaConfig
from privacy_estimates.experiments.aml import WorkspaceConfig
from privacy_estimates.experiments.challenge_point_selectors import TopKChallengePoints
from privacy_estimates.experiments.components import generate_canaries_with_secrets

from typing import Dict, Literal, Optional


EXPERIMENT_DIR = Path(__file__).parent


@dataclass
class CanaryConfig:
    method: str


@dataclass
class SharedTrainingParameters:
    base_model: str
    sequence_len: int
    learning_rate: float
    num_train_epochs: float
    lr_scheduler_type: str
    per_device_train_batch_size: int
    gradient_accumulation_steps: int


@dataclass
class SharedInferenceParameters:
    per_device_batch_size: int


class TrainTransformerComponentLoader(TrainingComponentLoader):
    def __init__(self, aml_component_loader: AMLComponentLoader, parameters: SharedTrainingParameters):
        super().__init__(aml_component_loader=aml_component_loader)
        self.parameters = parameters

    def load(self, train_data: Input, validation_data: Input, seed: int):
        component = self.aml_loader.load_from_component_spec(EXPERIMENT_DIR/"subpipelines"/"train.yml")
        job = component(**asdict(self.parameters), train_data=train_data, val_data=validation_data, seed=seed)
        job.component.jobs["train"].compute = self.aml_loader.workspace.gpu_compute
        return job


class TransformerInferenceComponentLoader(InferenceComponentLoader):
    def __init__(self, aml_component_loader: AMLComponentLoader, parameters: SharedInferenceParameters, mi_signal_method: str,
                 mi_signal_aggregation: str, mi_signal_extra_args: Optional[Dict] = None):
        super().__init__(aml_component_loader=aml_component_loader)
        self.parameters = parameters
        self.mi_signal_method = mi_signal_method
        self.mi_signal_agggregation = mi_signal_aggregation
        self.mi_signal_extra_args = mi_signal_extra_args or {}

    def load(self, model: Input, dataset: Input):
        component = self.aml_loader.load_from_component_spec(EXPERIMENT_DIR/"subpipelines"/"inference.yml")
        job = component(base_model=model, data=dataset, **asdict(self.parameters), mi_signal_method=self.mi_signal_method,
                        mi_signal_extra_args=" ".join(f"{k}={v}" for k, v in self.mi_signal_extra_args.items()),
                        mi_signal_aggregation=self.mi_signal_agggregation)
        job.component.jobs["inference"].compute = self.aml_loader.workspace.gpu_compute
        return job


class Game(BlackBoxMembershipInferenceGameBase):
    def __init__(self, shared_training_parameters: SharedTrainingParameters,
                 shared_inference_parameters: SharedInferenceParameters, workspace: WorkspaceConfig,
                 game_config: GameConfig, mi_signal_config: MISignalConfig, rmia_config: RmiaConfig,
                 shadow_model_config: ShadowModelConfig, canary_config: CanaryConfig) -> None:

        train_loader = TrainTransformerComponentLoader(
            aml_component_loader=AMLComponentLoader(workspace=workspace),
            parameters=shared_training_parameters
        )

        inference_loader = TransformerInferenceComponentLoader(
            aml_component_loader=AMLComponentLoader(workspace=workspace),
            parameters=shared_inference_parameters,
            mi_signal_method=mi_signal_config.method,
            mi_signal_extra_args=mi_signal_config.extra_args,
            mi_signal_aggregation=mi_signal_config.aggregation
        )

        attack_loader = RmiaLoader(offline_a=rmia_config.offline_a)

        challenge_point_selection_loader = TopKChallengePoints(
            num_challenge_points=game_config.num_challenge_points_per_model*game_config.num_models
        )

        self.canary_config = canary_config

        super().__init__(
            workspace=workspace,
            game_config=game_config,
            train_loader=train_loader,
            inference_loader=inference_loader,
            attack_loader=attack_loader,
            challenge_point_selection_loader=challenge_point_selection_loader,
            shadow_model_config=shadow_model_config
        )

    def preprocess_datasets(
        self
    ) -> Dict[Literal['train_data'] | Literal['validation_data'] | Literal['canary_data'], Input | Output]:
        train_data = self.workspace.ml_client.data.get(name="SST2-train", version="5")
        val_data = self.workspace.ml_client.data.get(name="SST2-test", version="5")
        if self.canary_config.method == "external_data":
            canary_data = self.workspace.ml_client.data.get(name="AmazonPolarity5k-train", version="2")
        elif self.canary_config.method == "generate_secrets":
            canary_data = generate_canaries_with_secrets(
                format="language_modelling", text_column="sentence", num_canaries=10000, seed=self.game_config.seed+230230
            ).outputs.output
        else:
            raise ValueError(f"Canary method {self.canary_config.method} not supported")
        return {"train_data": train_data, "validation_data": val_data, "canary_data": canary_data}


if __name__ == "__main__":
    Game.main(config_path=EXPERIMENT_DIR/"configs")
