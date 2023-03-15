import numpy as np
import matplotlib.pyplot as plt
from climatenet.utils.metrics import get_iou_perClass


def bootstrap_ious(cms_model1, cms_model2, B=10000):
    assert cms_model1.shape == cms_model2.shape

    diff_iou = []

    for i in range(B):
        # get bootstrap sample
        sample_indices = np.random.choice(list(range(len(cms_model1))), size=len(cms_model1), replace=True)
        cms1_bootstrap_sample = cms_model1[sample_indices]
        cms2_bootstrap_sample = cms_model2[sample_indices]

        # compute difference in iou between both models on bootstap sample
        aggregate_cm_boostrap1 = np.sum(cms1_bootstrap_sample, axis=0)
        aggregate_cm_boostrap2 = np.sum(cms2_bootstrap_sample, axis=0)

        boostrap_iou1 = get_iou_perClass(aggregate_cm_boostrap1)
        boostrap_iou2 = get_iou_perClass(aggregate_cm_boostrap2)

        boostrap_iou1 = get_iou_perClass(aggregate_cm_boostrap1)
        boostrap_iou2 = get_iou_perClass(aggregate_cm_boostrap2)

        diff_iou.append(boostrap_iou2 - boostrap_iou1)

    return np.stack(diff_iou)


def compute_bootstrap_pct_positive_diff(diff_iou):
    
    metrics = [
        (diff_iou[:, 0] > 0).mean(), # percentage of time that model 2 outperforms model 1 on the background class
        (diff_iou[:, 1] > 0).mean(),
        (diff_iou[:, 2] > 0).mean(),
        (diff_iou.mean(axis=1) > 0).mean()
    ]
    
    return metrics


def compute_bootstrap_confidence_intervals(diff_iou, alpha = 0.05):

    intervals = np.stack([
        np.quantile(diff_iou[:, 0], [alpha/2, 1-alpha/2]),
        np.quantile(diff_iou[:, 1], [alpha/2, 1-alpha/2]),
        np.quantile(diff_iou[:, 2], [alpha/2, 1-alpha/2]),
        np.quantile(diff_iou.mean(axis=1), [alpha/2, 1-alpha/2]),
    ])
    return intervals


def plot_bootstrap_results(diff_iou, figsize=(16, 3)):
    
    fig, axs = plt.subplots(1, 4, figsize=figsize)

    axs[0].hist(diff_iou[:, 0], density=True, bins=100)
    axs[0].set_title("Background Class")
    axs[0].set_ylabel("Frequency")
    axs[0].set_xlabel("Model 2 IOU - Model 1 IOU")

    axs[1].hist(diff_iou[:, 1], density=True, bins=100)
    axs[1].set_title("ARs")
    axs[1].set_ylabel("Frequency")
    axs[1].set_xlabel("Model 2 IOU - Model 1 IOU")

    axs[2].hist(diff_iou[:, 2], density=True, bins=100)
    axs[2].set_title("TCs")
    axs[2].set_ylabel("Frequency")
    axs[2].set_xlabel("Model 2 IOU - Model 1 IOU")

    axs[3].hist(diff_iou.mean(axis=1), density=True, bins=100)
    axs[3].set_title("Average Across Classes")
    axs[3].set_ylabel("Frequency")
    axs[3].set_xlabel("Model 2 IOU - Model 1 IOU")

    plt.tight_layout()
    plt.show()