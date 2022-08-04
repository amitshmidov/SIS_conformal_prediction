import os
import pickle

import numpy as np


def percent_greatest_sis_size_is_correct_label(experiment_dict):
    sis_size = experiment_dict["masks_sizes"]
    correctness = experiment_dict["corrects"]
    return np.sum(np.argmax(sis_size, axis=0) == np.argmax(correctness, axis=0)) / sis_size.shape[1]


def avg_sis_size_for_correct_and_wrong(experiment_dict):
    sis_size = experiment_dict["masks_sizes"]
    correctness = experiment_dict["corrects"].astype(bool)
    return np.mean(sis_size[0, :][correctness[0, :]]), np.mean(sis_size[0, :][~correctness[0, :]])


if __name__ == '__main__':
    for sis_file_name in os.listdir("../sufficient_input_subsets/approx_sis_results"):
        file_name = f"../sufficient_input_subsets/approx_sis_results/{sis_file_name}"
        with open(file_name, "rb") as results_file:
            experiment_dict = pickle.load(results_file)

        print(sis_file_name)
        print(percent_greatest_sis_size_is_correct_label(experiment_dict))
        print(avg_sis_size_for_correct_and_wrong(experiment_dict))
        print("\n")
