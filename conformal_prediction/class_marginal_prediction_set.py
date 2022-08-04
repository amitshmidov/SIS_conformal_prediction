import pickle

import numpy as np
import os
import torch
from torchvision import datasets

from util import data_util

# np.random.seed(42)


def load_sis(experiment_name):
    with open(f"../sufficient_input_subsets/approx_sis_results/{experiment_name}.pkl", "rb") as results_file:
        results_dict = pickle.load(results_file)
    num_images = results_dict["masks_sizes"].shape[1]
    sis = np.zeros((num_images, 10))
    for i in range(num_images):
        for rank in range(10):
            sis[i, results_dict["pred_labels"][rank, i].astype(int)] = results_dict["masks_sizes"][rank, i]
    return sis


def compute_quantiles(alpha, sis):
    n = sis.shape[0]
    Q = np.sort(sis, axis=0)[np.floor(alpha * (n + 1)).astype(int), :]
    print(f"Quantile: {Q}")
    return Q


def compute_prediction_set(alpha, calibration_sis, test_sis):
    Q = compute_quantiles(alpha, calibration_sis)
    return np.where(test_sis >= Q)[0]


if __name__ == '__main__':
    transform = data_util.cifar_test_transform()
    dataset = datasets.CIFAR10(root='data/',
                               train=False,
                               transform=transform,
                               download=True)

    experiment_name = "1000_images__015_threshold__absolute"
    sis = load_sis(experiment_name)
    test_image = np.random.choice(range(sis.shape[0]))
    calibration_sis = np.vstack([sis[:test_image, :], sis[test_image+1:, :]])
    test_sis = sis[test_image, :]
    print(f"Test image index: {test_image}")
    print(f"Test image SIS: {test_sis}")
    print(f"Prediction set: {compute_prediction_set(0.95, calibration_sis, test_sis)}")
    print(f"True label: {dataset[test_image][1]}")
