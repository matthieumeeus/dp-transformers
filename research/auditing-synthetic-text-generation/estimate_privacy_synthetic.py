from azure.ai.ml import Input, Output
from pathlib import Path
from dataclasses import dataclass, asdict

from privacy_estimates.experiments.loaders import InferenceComponentLoader, TrainingComponentLoader, AMLComponentLoader
from privacy_estimates.experiments.games.black_box_membership_inference import (
    BlackBoxMembershipInferenceGameBase, GameConfig, ShadowModelConfig
)
from privacy_estimates.experiments.attacks.rmia import RmiaLoader, RmiaConfig
from privacy_estimates.experiments.aml import WorkspaceConfig, ClusterComputeConfig, ComputeConfig, ServerlessComputeConfig
from privacy_estimates.experiments.challenge_point_selectors import TopKChallengePoints

from typing import Dict, Literal, Optional

from estimate_privacy_black_box_model_access import DataConfig, CanaryConfig, \
    ReplaceTokensComponentLoader, DataFilterComponentLoader, InDistributionCanaryComponentLoader, ExternalCanaryComponentLoader

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
    target_epsilon: float = float('inf')
    target_delta: float = 1.0
    per_sample_max_grad_norm: float = 1000.0

@dataclass
class SharedInferenceParameters:
    real_label_column: str
    real_text_column: str
    synthetic_label_column: str
    synthetic_text_column: str
    mia_method: str

class TrainTransformerComponentLoader(TrainingComponentLoader):
    def __init__(self, aml_component_loader: AMLComponentLoader, parameters: SharedTrainingParameters,
                  train_compute_config: ComputeConfig, generate_compute_config: ComputeConfig):
        super().__init__(aml_component_loader=aml_component_loader)
        self.parameters = parameters
        self.train_compute_config = train_compute_config
        self.generate_compute_config = generate_compute_config

    def load(self, train_data: Input, validation_data: Input, seed: int):
        params = asdict(self.parameters)
        if params["target_epsilon"] == float('inf'):
            params.pop("target_epsilon")
            params.pop("target_delta")
            params.pop("per_sample_max_grad_norm")
            component = self.aml_loader.load_from_component_spec(EXPERIMENT_DIR/"subpipelines"/"finetune_w_synthetic.yml")
        else:
            component = self.aml_loader.load_from_component_spec(EXPERIMENT_DIR/"subpipelines"/"finetune_w_synthetic_dp.yml")

        job = component(**params, train_data=train_data, val_data=validation_data, seed=seed)
        job.component.jobs["fine_tune"] = self.train_compute_config.apply(job.component.jobs["fine_tune"])
        job.component.jobs["generate"] = self.generate_compute_config.apply(job.component.jobs["generate"])
        return job


class TransformerInferenceComponentLoader(InferenceComponentLoader):
    def __init__(self, aml_component_loader: AMLComponentLoader, parameters: SharedInferenceParameters, compute_config: ComputeConfig):
        super().__init__(aml_component_loader=aml_component_loader)
        self.parameters = parameters
        self.compute_config = compute_config

    def load(self, model: Input, dataset: Input):
        component = self.aml_loader.load_from_component_spec(EXPERIMENT_DIR/"subpipelines"/"inference_synthetic.yml")
        job = component(synthetic_data=model, inference_data=dataset, 
                        **asdict(self.parameters))
        job.component.jobs["synthetic_membership_score"] = self.compute_config.apply(job.component.jobs["synthetic_membership_score"])
        return job

class Game(BlackBoxMembershipInferenceGameBase):
    def __init__(self, shared_training_parameters: SharedTrainingParameters,
                 shared_inference_parameters: SharedInferenceParameters, workspace: WorkspaceConfig,
                 game_config: GameConfig, rmia_config: RmiaConfig,
                 shadow_model_config: ShadowModelConfig, canary_config: CanaryConfig, data_config: DataConfig) -> None:
        
        #self.gpu_distributed_config = ServerlessComputeConfig(**workspace.compute['gpu_distributed_serverless'])
        #self.gpu_single_config = ServerlessComputeConfig(**workspace.compute['gpu_single_serverless'])
        self.gpu_distributed_config = ClusterComputeConfig(**workspace.compute['gpu_distributed'])
        self.gpu_single_config = ClusterComputeConfig(**workspace.compute['gpu_single'])

        train_loader = TrainTransformerComponentLoader(
            aml_component_loader=AMLComponentLoader(workspace=workspace),
            parameters=shared_training_parameters,
            train_compute_config=self.gpu_distributed_config,
            generate_compute_config=self.gpu_single_config
        )

        inference_loader = TransformerInferenceComponentLoader(
            aml_component_loader=AMLComponentLoader(workspace=workspace),
            parameters=shared_inference_parameters,
            compute_config=self.gpu_single_config
        )

        if 'ngram' in shared_inference_parameters.mia_method:
            use_log_column = True
        else:
            use_log_column = False

        attack_loader = RmiaLoader(offline_a=rmia_config.offline_a, use_log_column=use_log_column)

        challenge_point_selection_loader = TopKChallengePoints(
            num_challenge_points=game_config.num_challenge_points_per_model*game_config.num_models, allow_fewer=True
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

    @property
    def default_compute(self) -> ClusterComputeConfig:
        return ClusterComputeConfig(**self.workspace.compute['cpu'])
    
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
            in_distribution_canary_outputs = InDistributionCanaryComponentLoader(aml_component_loader=AMLComponentLoader(workspace=self.workspace), 
                                    text_name=self.data_config.text_column, canaries_min_words=self.canary_config.canary_length,
                                    n_canaries=self.canary_config.n_canaries, seed=self.canary_config.seed).load(train_data=train_data).outputs
            train_data = in_distribution_canary_outputs.updated_training_data
            canary_data = in_distribution_canary_outputs.canary_data

        elif self.canary_config.canary_method in ("sample_real", "sample_synthetic"):
            external_canary_outputs = ExternalCanaryComponentLoader(aml_component_loader=AMLComponentLoader(workspace=self.workspace), 
                                    canary_parameters=self.canary_config, train_parameters=self.train_config, compute_config=self.gpu_single_config).load(
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
