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
from privacy_estimates.experiments.components import generate_canaries_with_secrets, random_split_dataset, select_top_k_rows

from typing import Dict, Literal, Optional


EXPERIMENT_DIR = Path(__file__).parent

@dataclass
class DataConfig:
    train_data_name: str
    train_data_version: str
    eval_data_name: str
    eval_data_version: str
    min_words: int
    text_column: str

@dataclass
class CanaryConfig:
    method: str
    canary_min_words: int
    n_canaries: int
    canary_data_name: str
    canary_data_version: str
    canary_text_column: str
    num_tokens_to_replace: int
    replacement_method: str
    mlm_name: str

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

class DataFilterComponentLoader(TrainingComponentLoader):
    def __init__(self, aml_component_loader: AMLComponentLoader, min_words: int, text_column: str):
        super().__init__(aml_component_loader=aml_component_loader)
        self.min_words = min_words
        self.text_column = text_column

    def load(self, all_data: Input):
        component = self.aml_loader.load_from_component_spec(path=EXPERIMENT_DIR/"components"/"filter_data/component_spec.yml")
        job = component(all_data=all_data, min_words=self.min_words, text_column=self.text_column)
        return job
    
class ReplaceTokensComponentLoader(TrainingComponentLoader):
    def __init__(self, aml_component_loader: AMLComponentLoader, parameters: CanaryConfig):
        super().__init__(aml_component_loader=aml_component_loader)
        self.parameters = parameters

    def load(self, original_data: Input):
        component = self.aml_loader.load_from_component_spec(path=EXPERIMENT_DIR/"components"/"edit_canaries/component_spec.yml")
        job = component(original_data=original_data, text_column=self.parameters.canary_text_column, 
                        num_tokens_to_replace=self.parameters.num_tokens_to_replace, replacement_method=self.parameters.replacement_method,
                        model_name=self.parameters.mlm_name)
        return job

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

        if self.canary_config.method == "hold_out_original_data":
            # to do: also implement min canary words for this method
            data_split = random_split_dataset(dataset=train_data, split_1_size=self.canary_config.n_canaries, seed=self.game_config.seed)
            canary_data = data_split.outputs.dataset_1
            train_data = data_split.outputs.dataset_2

        elif self.canary_config.method == "external_data":
            canary_data = self.workspace.ml_client.data.get(name=self.canary_config.canary_data_name, version=self.canary_config.canary_data_version)
            canary_data = DataFilterComponentLoader(aml_component_loader=AMLComponentLoader(workspace=self.workspace), 
                                    min_words=self.canary_config.canary_min_words, text_column=self.canary_config.canary_text_column).load(all_data=canary_data).outputs.filtered_data
            canary_data = select_top_k_rows(data=canary_data, k=self.canary_config.n_canaries).outputs.output

        elif self.canary_config.method == "generate_secrets":
            canary_data = generate_canaries_with_secrets(
                format="language_modelling", text_column="sentence", num_canaries=self.canary_config.n_canaries, seed=self.game_config.seed+230230
            ).outputs.output

        else:
            raise ValueError(f"Canary method {self.canary_config.method} not supported")
        
        if self.canary_config.num_tokens_to_replace > 0:
            canary_data = ReplaceTokensComponentLoader(aml_component_loader=AMLComponentLoader(workspace=self.workspace), 
                                parameters=self.canary_config).load(original_data=canary_data).outputs.modified_data
    
        return {"train_data": train_data, "validation_data": val_data, "canary_data": canary_data}

if __name__ == "__main__":
    Game.main(config_path=EXPERIMENT_DIR/"configs")
