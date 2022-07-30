import os

import torch
from torchvision import datasets
from tqdm import tqdm

import inference_util
import sis_util
from config import DEVICE
from util import data_util
from sufficient_input_subsets import sis

import numpy as np


TOP_CLASSES_NUM = 5


def single_image_single_y_sis(model, image, pred_class, pred_confidence, sis_threshold):
    initial_mask = sis.make_empty_boolean_mask_broadcast_over_axis([3, 32, 32], 0)
    fully_masked_image = np.zeros((3, 32, 32), dtype='float32')

    pred_class = int(pred_class)
    pred_confidence = float(pred_confidence)
    # if pred_confidence < sis_threshold:
    #     return None
    sis_threshold *= pred_confidence

    f_class = sis_util.make_f_for_class(model, pred_class, batch_size=128, add_softmax=True)
    sis_result = sis.find_sis(
        f_class,
        sis_threshold,
        image.cpu().numpy(),
        initial_mask,
        fully_masked_image,
    )
    return sis_result


def compute_sis_on_all_classes(model, dataset, sis_threshold, start_idx, end_idx, sis_out_dir):
    for i in tqdm(range(start_idx, end_idx+1)):
        image, label = dataset[i]
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).to(DEVICE)
        with torch.no_grad():
            preds = torch.sort(torch.softmax(model(image.unsqueeze(0).to(DEVICE))[0], 0).to("cpu"))
            preds_confidences = preds[0][-TOP_CLASSES_NUM:].to(DEVICE)
            preds_classes = preds[1][-TOP_CLASSES_NUM:].to(DEVICE)

        for pred_class, pred_confidence in tqdm(list(zip(preds_classes, preds_confidences)), leave=False):
            sis_outfile = os.path.join(sis_out_dir, f"cifar10_test__image_{i}__class_{pred_class}.npz")
            if os.path.exists(sis_outfile):
                continue  # File already exists.
            sis_result = single_image_single_y_sis(model, image, pred_class, pred_confidence, sis_threshold)
            if sis_result is None:  # No SIS exists.
                continue
            sis_util.save_sis_result(sis_result, sis_outfile)


if __name__ == '__main__':
    sis_threshold = 0.95
    transform = data_util.cifar_test_transform()
    dataset = datasets.CIFAR10(root='data/',
                               train=False,
                               transform=transform,
                               download=True)
    sis_out_dir = "../saved_models/resnet18_single_epoch"
    model = inference_util.load_saved_model(sis_out_dir + "/")
    model.to(DEVICE)
    model.eval()

    start_idx, end_idx = 0, 100

    compute_sis_on_all_classes(model, dataset, sis_threshold, start_idx, end_idx, sis_out_dir)
