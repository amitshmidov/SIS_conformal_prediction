import os
import pickle

import numpy as np
import torch
from torchvision import datasets
from tqdm import tqdm

import inference_util
import sis_util
from config import DEVICE
from imagenet import imagenet_backselect
from imagenet.imagenet_backselect import BackselectResult
from sufficient_input_subsets import sis
from util import data_util

TOP_CLASSES_NUM = 5


def get_k_highest_val(x: torch.Tensor, k):
    top_values, top_indices = torch.topk(x, k)
    return top_values[..., -1], top_indices[..., -1]


def multi_image_single_y_approx_sis(model, images, labels, label_rank, sis_thresholds, remove_per_iter,
                                    original_prob_relative_list, max_iters=None):

    batched_gradient_sis_results = imagenet_backselect.run_gradient_backward_selection(
        images,
        model,
        remove_per_iter,
        label_rank=label_rank,
        max_iters=max_iters,
        add_random_noise=False,
        cuda=True)

    sis_results = {}
    for sis_i, sis_threshold in enumerate(sis_thresholds):
        masks_sizes = []
        corrects = []
        pred_labels = []
        for i in tqdm(range(len(images))):
            original_image = images[i]
            true_label = labels[i]

            # print('image idx: ', i)
            original_pred = model(original_image.unsqueeze(0).to(DEVICE)).detach().cpu()[0]
            # print(original_pred, original_pred.shape)
            original_pred = torch.nn.functional.softmax(original_pred, dim=0)
            label_conf, pred_label = get_k_highest_val(original_pred, label_rank)
            is_correct = bool(pred_label == true_label)
            # print('Original confidence: ', label_conf)
            # print('Original pred label: ', pred_label, dataset.classes[pred_label])
            # print('Is correct: ', is_correct)
            # print('True Class: ', dataset.classes[true_label])

            # if not original_prob_relative_list[sis_i]:
            #     if label_conf < sis_threshold:
            #         masks_sizes += [0]
            #         continue

            # if not is_correct or label_conf < 0.9:
            #     print('skipping\n\n')
            #     continue

            bs_result = batched_gradient_sis_results[i]

            # mask_after_iter = np.where(bs_result.confidences_over_backselect >= sis_threshold)[0][-1]
            threshold = (label_conf.numpy() * sis_threshold) if original_prob_relative_list[sis_i] else sis_threshold

            if bs_result.confidences_over_backselect[0, pred_label] < threshold:
                masks_sizes += [0]
                corrects += [is_correct]
                pred_labels += [pred_label.item()]
                continue

            mask_after_iter = np.where(bs_result.confidences_over_backselect[:, pred_label] >= threshold)[0][-1]

            # print('Mask after iteration: ', mask_after_iter)
            mask = torch.from_numpy(bs_result.mask_order >= mask_after_iter)

            fully_masked_image = torch.zeros(original_image.shape)
            masked_image = torch.where(mask, original_image, fully_masked_image)
            mask_size = int(mask.sum())
            mask_size_frac = mask_size / float(
                original_image.shape[1] * original_image.shape[2])
            # print('Num pixels unmasked:  %s (%.1f%%)' % (mask_size, mask_size_frac * 100.))

            # Confidence on masked image.
            preds = model(masked_image.unsqueeze(0).to(DEVICE))
            confidences = torch.nn.functional.softmax(preds, dim=1).detach().cpu().numpy()
            # print('Predicted confidence on masked image: ', confidences.max(), confidences.argmax())
            # print('\n\n')

            masks_sizes += [mask_size]
            corrects += [is_correct]
            pred_labels += [pred_label.item()]
        sis_results[str(sis_threshold)] = (masks_sizes, corrects, pred_labels)

    return sis_results

    # start_idx, end_idx = 0, 100
    #
    # compute_sis_on_all_classes(model, dataset, sis_threshold, start_idx, end_idx, sis_out_dir)

    # """Run Batched Gradient BackSelect"""
    # assert len(images.shape) == 4  # Check for batch dimension.
    #
    # # Model to eval mode.
    # model.eval()
    #
    # if cuda:
    #     images = images.to(DEVICE)
    #
    # # Initialize masks as all zeros.
    # masks = torch.zeros(images.shape[0], 1, images.shape[2], images.shape[3], device=DEVICE, requires_grad=True)
    # masks_history = torch.zeros(images.shape[0], 1, images.shape[2], images.shape[3], device=DEVICE, dtype=torch.int)
    #
    # # Compute initial predicted class.
    # softmax = torch.nn.Softmax(dim=1)
    # original_confidences = softmax(model(images))
    # original_pred_confidences, original_pred_classes = get_k_highest_val(original_confidences, label_rank)
    # # original_pred_confidences = original_confidences[..., pred_class]
    # # original_pred_classes = torch.from_numpy(np.ones_like(original_pred_confidences.detach().cpu()).astype("int64")) * \
    # #                         pred_class
    #
    # # Run backward selection.
    # confidences_history = []
    #
    # if max_iters is None:
    #     max_iters = int(np.ceil(np.prod(masks.shape[1:]) / float(remove_per_iter)))
    #
    # for i in tqdm(range(max_iters)):
    #     # Reset gradients.
    #     model.zero_grad()
    #     if masks.grad is not None:
    #         masks.grad.data.zero_()
    #
    #     # Compute masked inputs.
    #     masked_images = (1 - masks) * images
    #     if cuda:
    #         masked_images = masked_images.to(DEVICE)
    #
    #     # Compute confidences on masked images toward original predicted classes.
    #     confidences = softmax(model(masked_images))
    #     pred_confidences = confidences.cpu().gather(1, original_pred_classes.cpu().unsqueeze(1)).to(DEVICE)
    #
    #     # Compute gradients.
    #     torch.sum(pred_confidences).backward()
    #     assert masks.grad is not None
    #     grad_vals = masks.grad.detach()
    #     if add_random_noise:
    #         noise = (torch.randn(masks.shape) * (random_noise_variance ** 0.5))
    #         if cuda:
    #             noise = noise.to(DEVICE)
    #         grad_vals += noise
    #
    #     # Find optimal pixels to mask, excluding previously masked values.
    #     not_yet_masked_idxs_tuple = (1 - masks).flatten(start_dim=1).cpu().nonzero(as_tuple=True)
    #     # We remove the same number of features per image per iteration, so all
    #     # images have the same number of pixels remaining.
    #     grad_vals_not_yet_masked = grad_vals.cpu().flatten(start_dim=1)[not_yet_masked_idxs_tuple].\
    #         reshape(masks.shape[0], -1).to(DEVICE)
    #     num_pixels_remaining = int((1 - masks[0]).sum())
    #     _, to_mask_idxs_offset = torch.topk(grad_vals_not_yet_masked, min(remove_per_iter, num_pixels_remaining))
    #     # Remove offset from removing values for already masked pixels.
    #     not_yet_masked_idxs = not_yet_masked_idxs_tuple[1].reshape(masks.shape[0], -1)
    #     to_mask_idxs = not_yet_masked_idxs[
    #         torch.arange(masks.shape[0]).unsqueeze(1).expand(-1, to_mask_idxs_offset.shape[1]).flatten().cpu(),
    #         to_mask_idxs_offset.flatten().cpu(),
    #     ].reshape(masks.shape[0], -1).to(DEVICE)
    #     if DEVICE == torch.device("mps"):
    #         to_mask_idxs_mask = torch.zeros(masks.shape[0], masks.shape[2], masks.shape[3], device="cpu",
    #                                         dtype=torch.bool)
    #     else:
    #         to_mask_idxs_mask = torch.zeros(masks.shape[0], masks.shape[2], masks.shape[3], device=DEVICE,
    #                                         dtype=torch.bool)
    #     to_mask_idxs_mask.view(masks.shape[0], -1)[
    #         torch.arange(masks.shape[0]).unsqueeze(1).expand(-1, to_mask_idxs_offset.shape[1]).flatten(),
    #         to_mask_idxs.cpu().flatten(),
    #     ] = 1
    #     to_mask_idxs_mask = to_mask_idxs_mask.unsqueeze(1)  # Add broadcast over channels dimension.
    #     assert bool(torch.all(to_mask_idxs_mask.sum(dim=(2, 3)).flatten() == to_mask_idxs_offset.shape[1]))
    #     assert (to_mask_idxs_mask.cpu() + masks.cpu()).max() == 1
    #
    #     # Update mask and history.
    #     with torch.no_grad():
    #         masks.cpu()[to_mask_idxs_mask] = 1
    #         masks_history.cpu()[to_mask_idxs_mask] = i
    #
    #     confidences_history.append(confidences.detach().cpu().numpy())
    #
    # # Create BackselectResult objects.
    # confidences_history = np.array(confidences_history)  # For slicing.
    # batched_gradient_sis_results = []
    # for i in tqdm(range(images.shape[0])):
    #     batched_gradient_sis_results.append(BackselectResult(
    #         original_confidences=(
    #             original_confidences[i].detach().cpu().numpy()),
    #         target_class_idx=original_pred_classes[i].detach().cpu().numpy(),
    #         confidences_over_backselect=confidences_history[:, i, :],
    #         mask_order=masks_history[i].detach().cpu().numpy(),
    #     ))
    #
    # masks_sizes = []
    # for i in tqdm(range(len(images))):
    #     original_image = images[i]
    #
    #     original_pred = model(original_image.unsqueeze(0).to(DEVICE)).detach().cpu()[0]
    #     original_pred = torch.nn.functional.softmax(original_pred, dim=0)
    #
    #     bs_result = batched_gradient_sis_results[i]
    #
    #     confidences_over_backselect = bs_result.confidences_over_backselect
    #     mask_after_iter = np.where(confidences_over_backselect >= (original_pred.numpy() * sis_threshold))[0][-1]
    #     mask = torch.from_numpy(bs_result.mask_order >= mask_after_iter)
    #
    #     mask_size = int(mask.sum())
    #     masks_sizes += [mask_size]
    #
    # return masks_sizes


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
    for i in tqdm(range(start_idx, end_idx + 1)):
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


def find_sis_mask_from_backselect_result(bs_result, sis_threshold):
    mask_after_iter = np.where(
        bs_result.confidences_over_backselect >= sis_threshold)[0][-1]
    # print('Mask after iteration: ', mask_after_iter)
    mask = torch.from_numpy(bs_result.mask_order >= mask_after_iter)
    return mask


def generate_sis(num_images, sis_thresholds, original_prob_relative_list):
    transform = data_util.cifar_test_transform()
    dataset = datasets.CIFAR10(root='data/',
                               train=False,
                               transform=transform,
                               download=True)
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=num_images,
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=2)

    sis_out_dir = "../saved_models/resnet18_single_epoch"
    # sis_out_dir = "../saved_models/resnet18_rep1"
    model = inference_util.load_saved_model(sis_out_dir + "/")
    model.to(DEVICE)
    model.eval()

    remove_per_iter = 10
    max_iters = None
    images, labels = next(iter(data_loader))
    print("Data loaded")

    all_sis_all_ranks_results = {}
    for sis_threshold in sis_thresholds:
        all_sis_all_ranks_results[str(sis_threshold)] = {}
        all_sis_all_ranks_results[str(sis_threshold)]["all_masks_sizes"] = np.zeros((10, num_images))
        all_sis_all_ranks_results[str(sis_threshold)]["all_corrects"] = np.zeros((10, num_images))
        all_sis_all_ranks_results[str(sis_threshold)]["all_pred_labels"] = np.zeros((10, num_images))

    for label_rank in range(1, 11):
        sis_results = multi_image_single_y_approx_sis(model, images, labels, label_rank, sis_thresholds,
                                                      remove_per_iter, original_prob_relative_list, max_iters)
        for sis_i, sis_threshold in enumerate(sis_thresholds):
            masks_sizes, corrects, pred_labels = sis_results[str(sis_threshold)]
            print(f"Label rank: {label_rank}, SIS threshold: {sis_threshold}, "
                  f"Relative: {original_prob_relative_list[sis_i]}")
            print(masks_sizes)
            print(corrects)
            print(pred_labels)
            print("\n")

            all_sis_all_ranks_results[str(sis_threshold)]["all_masks_sizes"][(label_rank - 1), :] = masks_sizes
            all_sis_all_ranks_results[str(sis_threshold)]["all_corrects"][(label_rank - 1), :] = corrects
            all_sis_all_ranks_results[str(sis_threshold)]["all_pred_labels"][(label_rank - 1), :] = pred_labels
    # results = {"masks_sizes": all_masks_sizes, "corrects": all_corrects, "pred_labels": all_pred_labels}

    for sis_i, sis_threshold in enumerate(sis_thresholds):
        with open(f"../sufficient_input_subsets/approx_sis_results/"
                  f"{num_images}_images__"
                  f"{str(sis_threshold).replace('.', '')}_threshold__"
                  f"{'relative' if original_prob_relative_list[sis_i] else 'absolute'}.pkl", "wb") as results_file:
            pickle.dump({"masks_sizes": all_sis_all_ranks_results[str(sis_threshold)]["all_masks_sizes"],
                         "corrects": all_sis_all_ranks_results[str(sis_threshold)]["all_corrects"],
                         "pred_labels": all_sis_all_ranks_results[str(sis_threshold)]["all_pred_labels"]}, results_file)

    # sis_threshold = 0.5
    # threshold_confidence = original_pred
    # # threshold_confidence[pred_label] = label_conf * sis_threshold
    # threshold_confidence[pred_label] = sis_threshold
    # threshold_confidence = threshold_confidence.numpy()
    # threshold_confidence = sis_threshold
    # print(np.where(bs_result.confidences_over_backselect > threshold_confidence)[0][-1])


if __name__ == '__main__':
    num_images = 1000
    # num_images = 5

    sis_thresholds = [0.95, 0.9, 0.8, 0.5, 0.3, 0.15]
    # sis_thresholds = [0.95, 0.9, 0.8]
    original_prob_relative_list = [True, True, True, False, False, False]
    # original_prob_relative_list = [True, True, True]

    generate_sis(num_images, sis_thresholds, original_prob_relative_list)
    # generate_sis(num_images, [0.95], [True])
    # generate_sis(num_images, [0.9], [True])
    # generate_sis(num_images, [0.8], [True])
    #
    # generate_sis(num_images, [0.5], [False])
    # generate_sis(num_images, [0.3], [False])
    # generate_sis(num_images, [0.15], [False])

