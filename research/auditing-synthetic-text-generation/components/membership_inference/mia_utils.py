import random
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score

def select_samples(dataset, text_name, number_of_samples):    
    selected_indices =  random.sample(range(len(dataset)), number_of_samples)
    samples = dataset.select(selected_indices)[text_name]
    return samples

def compute_tpr_at_fpr(y_true, y_prob, target_fpr):
    """
    Compute the true positive rate (TPR) at a given false positive rate (FPR).

    Parameters:
    y_true (np.array): Array of true binary labels (0 or 1).
    y_prob (np.array): Array of predicted probabilities.
    target_fpr (float): The target false positive rate at which to compute the TPR.

    Returns:
    float: The true positive rate at the target false positive rate.
    """
    # Compute FPR, TPR, and thresholds
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    
    # Interpolate to find the TPR at the desired FPR
    tpr_at_target_fpr = np.interp(target_fpr, fpr, tpr)
    
    return tpr_at_target_fpr

def compute_performance(scores_members, scores_non_members):
    all_mia_performance = {}
    
    for method in scores_members.keys():
        mia_performance = {}
        # make sure to take the negative score - as loss/distance is expected to be smaller for members (positive class)
        member_vals = [-val for val in scores_members[method]]
        non_member_vals = [-val for val in scores_non_members[method]]
        mia_performance['auc'] = roc_auc_score([1]*len(member_vals) + [0]*len(non_member_vals), member_vals + non_member_vals)
        for fpr in (0.01, 0.05, 0.1):
            mia_performance[f'tpr_at_{fpr}'] = compute_tpr_at_fpr([1]*len(member_vals) + [0]*len(non_member_vals), member_vals + non_member_vals, fpr)
        print(f"Method: {method}, AUC: {mia_performance['auc']}, TPR@0.01: {mia_performance['tpr_at_0.01']}, TPR@0.05: {mia_performance['tpr_at_0.05']}, TPR@0.1: {mia_performance['tpr_at_0.1']}")
        all_mia_performance[method] = mia_performance

    return all_mia_performance