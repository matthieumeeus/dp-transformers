import pandas as pd
import numpy as np
import os
from privacy_estimates.experiments.aml import Job
from datasets import Dataset
from tempfile import TemporaryDirectory
from sklearn.metrics import roc_curve
import warnings
from typing import Mapping, Hashable, Optional, Sequence

results = Dataset.from_pandas(pd.DataFrame(data=[
    {
        "dataset": "sst",
        "canary_method": "synth",
        "canary_label": "can",
        "mi_method": "model",
        "eps": 8.0,
        "url": "https://ml.azure.com/experiments/id/cbd45cd3-4fd8-4922-b82a-124527cc98ee/runs/gentle_crowd_zd5rx64gg2?wsid=/subscriptions/acc09744-1ee3-4242-b375-93421c63af0c/resourcegroups/PPML/providers/Microsoft.MachineLearningServices/workspaces/M365Research-PPML-EUS&tid=72f988bf-86f1-41af-91ab-2d7cd011db47"
    },
    {
        "dataset": "sst", 
        "canary_method": "synth",
        "canary_label": "uni",
        "mi_method": "model",
        "eps": 8.0,
        "url": "https://ml.azure.com/experiments/id/cbd45cd3-4fd8-4922-b82a-124527cc98ee/runs/olden_square_bmz8fc5pd3?wsid=/subscriptions/acc09744-1ee3-4242-b375-93421c63af0c/resourcegroups/PPML/providers/Microsoft.MachineLearningServices/workspaces/M365Research-PPML-EUS&tid=72f988bf-86f1-41af-91ab-2d7cd011db47"
    },
    {
        "dataset": "snli",
        "canary_method": "in",
        "canary_label": None,
        "mi_method": "model",
        "eps": 8.0,
        "url": "https://ml.azure.com/experiments/id/cbd45cd3-4fd8-4922-b82a-124527cc98ee/runs/dynamic_tangelo_pq7lbck108?wsid=/subscriptions/acc09744-1ee3-4242-b375-93421c63af0c/resourcegroups/PPML/providers/Microsoft.MachineLearningServices/workspaces/M365Research-PPML-EUS&tid=72f988bf-86f1-41af-91ab-2d7cd011db47"
    },
    {
        "dataset": "agnews", 
        "canary_method": "synth",
        "canary_label": "can",
        "mi_method": "model",
        "eps": 8.0,
        "url": "https://ml.azure.com/experiments/id/cbd45cd3-4fd8-4922-b82a-124527cc98ee/runs/quirky_cheetah_64wynrjcf5?wsid=/subscriptions/acc09744-1ee3-4242-b375-93421c63af0c/resourcegroups/PPML/providers/Microsoft.MachineLearningServices/workspaces/M365Research-PPML-EUS&tid=72f988bf-86f1-41af-91ab-2d7cd011db47"
    },
    {
        "dataset": "snli",
        "canary_method": "synth",
        "canary_label": "uni",
        "mi_method": "model",
        "eps": 8.0,
        "url": "https://ml.azure.com/experiments/id/cbd45cd3-4fd8-4922-b82a-124527cc98ee/runs/happy_seed_ml4850vkgn?wsid=/subscriptions/acc09744-1ee3-4242-b375-93421c63af0c/resourcegroups/PPML/providers/Microsoft.MachineLearningServices/workspaces/M365Research-PPML-EUS&tid=72f988bf-86f1-41af-91ab-2d7cd011db47"
    },
    {
        "dataset": "sst",
        "canary_method": "in",
        "canary_label": None,
        "mi_method": "model",
        "eps": 8.0,
        "url": "https://ml.azure.com/experiments/id/cbd45cd3-4fd8-4922-b82a-124527cc98ee/runs/amiable_heart_77qz86tj31?wsid=/subscriptions/acc09744-1ee3-4242-b375-93421c63af0c/resourcegroups/PPML/providers/Microsoft.MachineLearningServices/workspaces/M365Research-PPML-EUS&tid=72f988bf-86f1-41af-91ab-2d7cd011db47",
    },
    {
        "dataset": "agnews",
        "canary_method": "in",
        "canary_label": None,
        "mi_method": "model",
        "eps": 8.0,
        "url": "https://ml.azure.com/experiments/id/cbd45cd3-4fd8-4922-b82a-124527cc98ee/runs/mighty_star_qssbkwy1hh?wsid=/subscriptions/acc09744-1ee3-4242-b375-93421c63af0c/resourcegroups/PPML/providers/Microsoft.MachineLearningServices/workspaces/M365Research-PPML-EUS&tid=72f988bf-86f1-41af-91ab-2d7cd011db47",
    },
    {
        "dataset": "agnews",
        "canary_method": "synth",
        "canary_label": "uni",
        "mi_method": "model",
        "eps": 8.0,
        "url": "https://ml.azure.com/experiments/id/cbd45cd3-4fd8-4922-b82a-124527cc98ee/runs/plucky_hand_p72zm58p3b?wsid=/subscriptions/acc09744-1ee3-4242-b375-93421c63af0c/resourcegroups/PPML/providers/Microsoft.MachineLearningServices/workspaces/M365Research-PPML-EUS&tid=72f988bf-86f1-41af-91ab-2d7cd011db47",
    },
    {
        "dataset": "snli",
        "canary_method": "synth",
        "canary_label": "can",
        "mi_method": "model",
        "eps": 8.0,
        "url": "https://ml.azure.com/experiments/id/cbd45cd3-4fd8-4922-b82a-124527cc98ee/runs/careful_gold_87wlh2yvcf?wsid=/subscriptions/acc09744-1ee3-4242-b375-93421c63af0c/resourcegroups/PPML/providers/Microsoft.MachineLearningServices/workspaces/M365Research-PPML-EUS&tid=72f988bf-86f1-41af-91ab-2d7cd011db47"
    }
]))

class RMIA:
    def __init__(
        self, reference_signals_out: Mapping[Hashable, float],
        reference_signals_in: Optional[Mapping[Hashable, float]] = None,
        offline_a: Optional[float] = None
    ):
        """
        Signals are probabilities P(x|\theta) and are checked to be in [0, 1].
        If they're not in [0,1] a warning is produced but the attack should still work (albeit more heuristically)
        provided that the convention is followed that larger signal represent evidence for in-membership.

        Args:
            reference_signals_out: Reference signals for out-of-distribution samples. A dictionary
                mapping sample indices to reference signals.
            reference_signals_in: Reference signals for in-distribution samples. A dictionary
                mapping sample indices to reference signals. If None, mean_in is computed using
                reference_signals_out.
            offline_a: Offline value for a. If provided, mean_in is computed using this value.
                If not provided, mean_in is computed using reference signals.
        """
        if (reference_signals_in is None) == (offline_a is None):
            raise ValueError("Either reference_signals_in or offline_a must be provided, but not both.")

    
        if any((v<0 or 1<v) for v in reference_signals_out.values()):
            warnings.warn(f"In-reference signals must be in the range [0, 1].", RuntimeWarning)
        if reference_signals_in is not None:
            if any((v<0 or 1<v) for v in reference_signals_in.values()):
                warnings.warn(f"Out-reference signals must be in the range [0, 1].", RuntimeWarning)

        if offline_a is not None:
            print("RMIA: Using offline_a for computing mean_in.")
        else:
            print("RMIA: Using reference signals for both in and out.")

        self.offline_a = offline_a
        self.reference_signals_out = reference_signals_out
        self.reference_signals_in = reference_signals_in

    @classmethod
    def from_dataset(cls, reference_signals_ds: Dataset, offline_a: Optional[float] = None, use_log_column: bool = False):
        required_columns = {"sample_index", "split"}
        if use_log_column:
            required_columns |= {"log_mi_signal_log_mean_exp_in", "log_mi_signal_log_mean_exp_out"}
        else:
            required_columns |= {"mi_signal_mean_in", "mi_signal_mean_out"}
        if not required_columns.issubset(reference_signals_ds.column_names):
            raise ValueError(
                f"Reference signals dataset must contain columns: {required_columns}. "
                f"Found columns: {reference_signals_ds.column_names}. "
                f"Missing columns: {required_columns - set(reference_signals_ds.column_names)}."
            )

        if use_log_column:
            signal_column_out = np.exp(np.array(reference_signals_ds["log_mi_signal_log_mean_exp_out"], dtype=np.longdouble))
            signal_column_in = np.exp(np.array(reference_signals_ds["log_mi_signal_log_mean_exp_in"], dtype=np.longdouble))
        else:
            signal_column_out = reference_signals_ds["mi_signal_mean_out"]
            signal_column_in = reference_signals_ds["mi_signal_mean_in"]

        reference_signals_out = {
            (split, sample_index): mi_signal
            for split, sample_index, mi_signal
            in zip(
                reference_signals_ds["split"], reference_signals_ds["sample_index"],
                signal_column_out
            )
        }
        if sum(reference_signals_ds["mi_signal_count_in"]) == 0:
            reference_signals_in = None
        else:
            reference_signals_in = {
                (split, sample_index): mi_signal
                for split, sample_index, mi_signal
                in zip(
                    reference_signals_ds["split"], reference_signals_ds["sample_index"],
                    signal_column_in
                )
            }
        return RMIA(offline_a=offline_a, reference_signals_out=reference_signals_out,
                    reference_signals_in=reference_signals_in)
	
    def compute_score(self, index: Sequence[Hashable], target_signals: Sequence[float]) -> np.ndarray:
        breakpoint()
        target_signals = np.array(target_signals)
        reference_signals_in = np.array([self.reference_signals_in[i] for i in index])
        assert len(reference_signals_in) == len(target_signals)
        assert len(target_signals) == len(index)
        mean_out_x = reference_signals_in # we have already computed the mean
        if self.offline_a:
            mean_x = ((1 + self.offline_a) / 2 * mean_out_x + (1 - self.offline_a) / 2)
        else:
            reference_signals_out = np.array([self.reference_signals_out[i] for i in index])
            mean_x = (reference_signals_in + reference_signals_out) / 2
        prob_ratio_x = target_signals.ravel() / mean_x
        mia_scores = prob_ratio_x
        return mia_scores


from safetensors import safe_open 


def compare_models(url: str):
    job = Job.from_url(url)
    train_ref_models = job.get_node("compute_shadow_model_statistics").get_node("train_shadow_models").get_node("train_final_model_group")
    train_ref_models_0 = train_ref_models.get_node("train_model_and_predict").get_node("train").get_node("fine_tune")
    train_ref_models_1 = train_ref_models.get_node("train_model_and_predict_1").get_node("train").get_node("fine_tune")

    with TemporaryDirectory() as tmpdir0:
        with TemporaryDirectory() as tmpdir1:
            train_ref_models_0.download_output("output_dir", path=tmpdir0)
            train_ref_models_1.download_output("output_dir", path=tmpdir1)
            with safe_open(os.path.join(tmpdir0, "adapter_model.safetensors"), framework="pt") as f0:
                with safe_open(os.path.join(tmpdir1, "adapter_model.safetensors"), framework="pt") as f1:
                    for key0, key1 in zip(f0.keys(), f1.keys()):
                        if key0 != key1:
                            raise ValueError(f"Keys do not match: {key0} != {key1}")
                        tensor0 = f0.get_tensor(key0)
                        tensor1 = f1.get_tensor(key1)
                        if not np.allclose(tensor0, tensor1, rtol=1e-5, atol=1e-8):
                            raise ValueError(f"Tensors do not match for key {key0}: {tensor0} != {tensor1}")

compare_models(url=results["url"][0])


def get_metrics(url: str):
    job = Job.from_url(url)
    attack = job.get_node("attack")
    estimates = job.get_node("estimate_privacy")
    with TemporaryDirectory() as tmpdir:
        estimates.download_input(name="scores", path=tmpdir)
        scores = Dataset.load_from_disk(str(tmpdir))
    with TemporaryDirectory() as tmpdir:
        estimates.download_input(name="challenge_bits", path=tmpdir)
        challenge_bits = Dataset.load_from_disk(str(tmpdir))
    with TemporaryDirectory() as tmpdir:
        attack.download_input(name="challenge_points", path=tmpdir)
        challenge_signal = Dataset.load_from_disk(str(tmpdir))
    with TemporaryDirectory() as tmpdir:
        attack.download_input(name="mi_statistics", path=tmpdir)
        mi_statistics = Dataset.load_from_disk(str(tmpdir))

    breakpoint()

    attack = RMIA.from_dataset(
        reference_signals_ds=mi_statistics, use_log_column=True,
    )

    target_indices = list(zip(challenge_signal["split"], challenge_signal["sample_index"]))
    target_signals = np.exp(np.array(challenge_signal["log_mi_signal"], dtype=np.longdouble))
 
    y_scores = attack.compute_score(index=target_indices, target_signals=target_signals)
    y_true = np.array(challenge_bits["challenge_bit"], dtype=np.longdouble)

    fpr, tpr, _ = roc_curve(y_true, y_scores)

    tpr_0_01 = np.interp(0.01, fpr, tpr)
    tpr_0_1 = np.interp(0.1, fpr, tpr)
    auc = np.trapz(tpr, fpr)
    return {"fpr": fpr, "tpr": tpr, "auc": auc, "tpr@0.01": tpr_0_01, "tpr@0.1": tpr_0_1}

results = results.map(get_metrics, input_columns=["url"])

print(results.remove_columns(["url", "fpr", "tpr"]).to_pandas())
