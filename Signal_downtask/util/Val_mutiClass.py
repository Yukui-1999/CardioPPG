import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample
import argparse
import scipy.stats as stats
import os
from tqdm import tqdm
from sklearn.preprocessing import label_binarize
def bootstrap_auc(y_true, y_pred, n_bootstrap=1000):
    aucs = []
    for i in tqdm(range(n_bootstrap)):
        indices = np.random.choice(len(y_true), len(y_true), replace=True)
        aucs.append(roc_auc_score(y_true[indices], y_pred[indices]))
    aucs = np.array(aucs)
    return np.percentile(aucs, [2.5, 97.5])

def calculate_confusion_matrix(y_true_binary, y_pred_binary):
    TP = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
    FP = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
    TN = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
    FN = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
    return TP, FP, TN, FN


def calculate_f1(TP, FP, FN):
    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    return f1


def simulate_f1(TP, FP, TN, FN, n_simulations=1000):
    f1_scores = []

    for _ in tqdm(range(n_simulations)):
        sim_TP = np.random.binomial(TP + FN, TP / (TP + FN)) if (TP + FN) > 0 else 0
        sim_FN = TP + FN - sim_TP
        
        # 检查 TP + FP 是否为 0，避免除零错误
        if (TP + FP) > 0:
            prob_FP = FP / (TP + FP)
        else:
            prob_FP = 0  # 如果 TP + FP 为 0，设置概率为 0
        
        sim_FP = np.random.binomial(TP + FP, prob_FP)
        
        sim_f1 = calculate_f1(sim_TP, sim_FP, sim_FN)
        f1_scores.append(sim_f1)


    return np.percentile(f1_scores, [2.5, 97.5])

def wilson_score(p, n):
        z = stats.norm.ppf(0.975)
        denominator = 1 + z**2 / n
        center = (p + z**2 / (2 * n)) / denominator
        margin = z * ((p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5) / denominator
        return (center - margin, center + margin)

def calculate_ovr_metrics(y_true, y_pred):
    n_classes = y_true.shape[1]
    metrics = {}
    
    for i in range(n_classes):
        y_true_binary = y_true[:, i]
        y_pred_binary = np.round(y_pred[:, i])  # Convert softmax probabilities to binary predictions for this class
        
        TP, FP, TN, FN = calculate_confusion_matrix(y_true_binary, y_pred_binary)
        sensitivity = TP / (TP + FN) if (TP + FN) != 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) != 0 else 0
        accuracy = (TP + TN) / (TP + FP + TN + FN) if (TP + FP + TN + FN) != 0 else 0
        PPV = TP / (TP + FP) if (TP + FP) != 0 else 0
        NPV = TN / (TN + FN) if (TN + FN) != 0 else 0
        F1 = 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) != 0 else 0
        total = TP + FP + TN + FN
        pe = ((TP + FN) * (TP + FP) + (FN + TN) * (FP + TN)) / (total ** 2) if total != 0 else 0
        kappa = (accuracy - pe) / (1 - pe) if (1 - pe) != 0 else 0
        
        # Compute Wilson Score CIs
        sensitivity_ci = wilson_score(sensitivity, TP + FN)
        specificity_ci = wilson_score(specificity, TN + FP)
        accuracy_ci = wilson_score(accuracy, total)
        PPV_ci = wilson_score(PPV, TP + FP) if (TP + FP) != 0 else (0, 0)
        NPV_ci = wilson_score(NPV, TN + FN) if (TN + FN) != 0 else (0, 0)
        F1_ci = simulate_f1(TP, FP, TN, FN)
        
        # Compute AUC and bootstrap CI
        auc = roc_auc_score(y_true_binary, y_pred[:, i])
        auc_ci = bootstrap_auc(y_true_binary, y_pred[:, i])

        # Store metrics for this class
        metrics[f"Class_{i}"] = {
            "Sensitivity": (sensitivity, sensitivity_ci),
            "Specificity": (specificity, specificity_ci),
            "Accuracy": (accuracy, accuracy_ci),
            "AUC": (auc, auc_ci),
            "PPV": (PPV, PPV_ci),
            "NPV": (NPV, NPV_ci),
            "F1-Score": (F1, F1_ci),
            "Kappa": kappa
        }

    return metrics