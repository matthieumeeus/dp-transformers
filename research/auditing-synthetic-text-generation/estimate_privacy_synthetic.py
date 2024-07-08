from azure.ai.ml import Input, Output
from pathlib import Path
from dataclasses import dataclass, asdict

from privacy_estimates.experiments.loaders import InferenceComponentLoader, TrainingComponentLoader, AMLComponentLoader
from privacy_estimates.experiments.games.black_box_membership_inference import (
    BlackBoxMembershipInferenceGameBase, GameConfig, ShadowModelConfig
)
from privacy_estimates.experiments.attacks.rmia import RmiaLoader, RmiaConfig
from privacy_estimates.experiments.aml import WorkspaceConfig
from privacy_estimates.experiments.challenge_point_selectors import TopKChallengePoints
from privacy_estimates.experiments.components import random_split_dataset

from typing import Dict, Literal, Optional

from estimate_privacy_black_box_model_access import DataConfig, CanaryConfig, \
    ReplaceTokensComponentLoader, DataFilterComponentLoader, ExternalCanaryComponentLoader

EXPERIMENT_DIR = Path(__file__).parent

@dataclass
class SharedTrainingParameters:
    model_path: Path
    templated_prompt: str
    text_column: str
    label_column: str
    sequence_len: int
    learning_rate: float
    num_train_epochs: float
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    enable_lora: bool
    lora_dim: int
    target_modules: str
    gradient_checkpointing: bool
    torch_dtype: str
    quantization_4bit: bool
    synthetic_multiple: int

@dataclass
class SharedInferenceParameters:
    real_label_column: str
    real_text_column: str
    synthetic_label_column: str
    synthetic_text_column: str
    mia_method: str

class TrainTransformerComponentLoader(TrainingComponentLoader):
    def __init__(self, aml_component_loader: AMLComponentLoader, parameters: SharedTrainingParameters):
        super().__init__(aml_component_loader=aml_component_loader)
        self.parameters = parameters

    def load(self, train_data: Input, validation_data: Input, seed: int):
        component = self.aml_loader.load_from_component_spec(EXPERIMENT_DIR/"subpipelines"/"finetune_w_synthetic.yml")
        job = component(**asdict(self.parameters), train_data=train_data, val_data=validation_data, seed=seed)
        job.component.jobs["fine_tune"].compute = self.aml_loader.workspace.gpu_compute
        job.component.jobs["generate"].compute = self.aml_loader.workspace.gpu_compute
        return job

class TransformerInferenceComponentLoader(InferenceComponentLoader):
    def __init__(self, aml_component_loader: AMLComponentLoader, parameters: SharedInferenceParameters):
        super().__init__(aml_component_loader=aml_component_loader)
        self.parameters = parameters

    def load(self, model: Input, dataset: Input):
        component = self.aml_loader.load_from_component_spec(EXPERIMENT_DIR/"subpipelines"/"inference_synthetic.yml")
        job = component(synthetic_data=model, inference_data=dataset, 
                        **asdict(self.parameters))
        job.component.jobs["synthetic_membership_score"].compute = self.aml_loader.workspace.gpu_compute
        return job

class Game(BlackBoxMembershipInferenceGameBase):
    def __init__(self, shared_training_parameters: SharedTrainingParameters,
                 shared_inference_parameters: SharedInferenceParameters, workspace: WorkspaceConfig,
                 game_config: GameConfig, rmia_config: RmiaConfig,
                 shadow_model_config: ShadowModelConfig, canary_config: CanaryConfig, data_config: DataConfig) -> None:

        train_loader = TrainTransformerComponentLoader(
            aml_component_loader=AMLComponentLoader(workspace=workspace),
            parameters=shared_training_parameters
        )

        inference_loader = TransformerInferenceComponentLoader(
            aml_component_loader=AMLComponentLoader(workspace=workspace),
            parameters=shared_inference_parameters
        )

        attack_loader = RmiaLoader(offline_a=rmia_config.offline_a)

        challenge_point_selection_loader = TopKChallengePoints(
            num_challenge_points=game_config.num_challenge_points_per_model*game_config.num_models
        )

        self.canary_config = canary_config
        self.data_config = data_config
        self.train_config = shared_training_parameters

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

        train_data = self.workspace.ml_client.data.get(name=self.data_config.train_data_name, version=self.data_config.train_data_version)
        val_data = self.workspace.ml_client.data.get(name=self.data_config.eval_data_name, version=self.data_config.eval_data_version)

        # filter the data
        train_data = DataFilterComponentLoader(aml_component_loader=AMLComponentLoader(workspace=self.workspace), 
                            min_words=self.data_config.min_words, text_column=self.data_config.text_column).load(all_data=train_data).outputs.filtered_data
        val_data = DataFilterComponentLoader(aml_component_loader=AMLComponentLoader(workspace=self.workspace), 
                            min_words=self.data_config.min_words, text_column=self.data_config.text_column).load(all_data=val_data).outputs.filtered_data

        if self.canary_config.canary_method == "hold_out_original_data":
            # to do: also implement min canary words for this method
            data_split = random_split_dataset(dataset=train_data, split_1_size=self.canary_config.n_canaries, seed=self.game_config.seed)
            canary_data = data_split.outputs.dataset_1
            train_data = data_split.outputs.dataset_2

        elif self.canary_config.canary_method in ("sample_real", "sample_synthetic"):
            external_canary_outputs = ExternalCanaryComponentLoader(aml_component_loader=AMLComponentLoader(workspace=self.workspace), 
                                    canary_parameters=self.canary_config, train_parameters=self.train_config).load(
                                        original_dataset=train_data, 
                                        external_artifact=self.workspace.ml_client.data.get(name=self.canary_config.external_artifact,
                                                                                            version=self.canary_config.external_artifact_version),
                                    ).outputs
            canary_data = external_canary_outputs.canary_dataset
            train_data = external_canary_outputs.updated_training_dataset

        else:
            raise ValueError(f"Canary method {self.canary_config.canary_method} not supported")
        
        if self.canary_config.num_tokens_to_replace > 0:
            canary_data = ReplaceTokensComponentLoader(aml_component_loader=AMLComponentLoader(workspace=self.workspace), 
                                parameters=self.canary_config).load(original_data=canary_data).outputs.modified_data
    
        return {"train_data": train_data, "validation_data": val_data, "canary_data": canary_data}

if __name__ == "__main__":
    Game.main(config_path=EXPERIMENT_DIR/"configs")
