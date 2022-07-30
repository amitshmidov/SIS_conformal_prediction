import numpy as np
import os
import torch
from torchvision import datasets

from util import data_util


def get_image_from_sis_file_name(sis_file_name):
    image_i = sis_file_name.split("__")[1].split("_")[1]
    return int(image_i)


def get_class_from_sis_file_name(sis_file_name):
    class_i = sis_file_name.split("__")[2].split("_")[1].split(".")[0]
    return int(class_i)


def load_sis(experiment_name):
    sis_files = [f for f in os.listdir(f"../saved_models/{experiment_name}/") if "npz" in f]
    images_idx = [get_image_from_sis_file_name(file_name) for file_name in sis_files]
    classes_idx = [get_class_from_sis_file_name(file_name) for file_name in sis_files]
    sis = np.zeros((np.max(images_idx) + 1, np.max(classes_idx) + 1))

    for image_i in images_idx:
        for class_i in classes_idx:
            file_name = f"cifar10_test__image_{image_i}__class_{class_i}.npz"
            full_file_name = f"../saved_models/{experiment_name}/{file_name}"
            if file_name in sis_files:
                sis[image_i, class_i] = np.load(full_file_name)["mask"].sum()
            else:
                sis[image_i, class_i] = 0
    return sis


# def compute_quantiles(X, Y, alpha, tau, experiment_name):
#     n = X.shape[0]
#     sis_tau = load_sis(experiment_name)
#     Q = np.sort(sis_tau, axis=0)[np.floor(alpha * (n + 1))]  # vector of Q_y-s for each y value
#
#
# def compute_prediction_set(x_test, Y, tau, alpha, experiment_name):
#     Q = compute_quantiles(X, Y, alpha, tau, experiment_name)


# if __name__ == '__main__':
    # transform = data_util.cifar_test_transform()
    # dataset = datasets.CIFAR10(root='data/',
    #                            train=False,
    #                            transform=transform,
    #                            download=True)
