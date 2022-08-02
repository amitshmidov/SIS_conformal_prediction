
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.utils as utils

from config import DEVICE
from imagenet import imagenet_backselect

import inference_util
from util import data_util


def resnet_try():
    # model = inference_util.load_saved_model('./saved_models/resnet18_rep1/')
    model = inference_util.load_saved_model('./saved_models/resnet18_single_epoch/')
    # model = resnet18()
    model.to(DEVICE)
    model.eval()
    print('Loaded model')

    # Load dataset.

    transform = data_util.cifar_test_transform()  # No augmentation
    dataset = datasets.CIFAR10(root='data/',
                               train=True,
                               transform=transform,
                               download=True)

    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])
    #
    # val_transform = transforms.Compose([
    #     transforms.Resize(256),
    #     transforms.CenterCrop(224),
    #     transforms.ToTensor(),
    #     normalize,
    # ])

    # val_data = datasets.ImageNet(IMAGENET_DIR, split='val',
    #                              transform=val_transform)
    # print('# val images: ', len(val_data))

    # val_loader = torch.utils.data.DataLoader(val_data,
    #                                          batch_size=32,
    #                                          shuffle=True,
    #                                          num_workers=4,
    #                                          pin_memory=True)

    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=5,
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=2)

    # Pre-trained ResNet-50 val accuracy:
    # print(accuracy(val_loader, resnet))

    images, labels = next(iter(data_loader))

    def show(img):
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
        plt.xticks([])
        plt.yticks([])

    def find_sis_mask_from_backselect_result(bs_result, sis_threshold):
        mask_after_iter = np.where(
            bs_result.confidences_over_backselect >= sis_threshold)[0][-1]
        # print('Mask after iteration: ', mask_after_iter)
        mask = torch.from_numpy(bs_result.mask_order >= mask_after_iter)
        return mask

    plt.figure(figsize=(15, 10))
    grid = utils.make_grid(images, nrow=8, padding=10, normalize=True)
    show(grid)
    plt.show()

    # Run Batched Gradient BackSelect on batch of images.

    images, labels = next(iter(data_loader))

    label_rank = 2

    batched_gradient_sis_results = imagenet_backselect.run_gradient_backward_selection(
        images,
        model,
        10,  # Remove k=100 pixels per iteration
        label_rank=label_rank,
        max_iters=None,
        add_random_noise=False,
        cuda=True)
    #%%
    # Generate SIS for val images that are correctly/confidently classified.
    # Skip images that are not predicted >= threshold initially.

    SIS_THRESHOLD = 0.1

    for i in range(len(images)):
        original_image = images[i]
        true_label = labels[i]

        print('image idx: ', i)
        original_pred = model(original_image.unsqueeze(0).to(DEVICE)).detach().cpu()[0]
        print(original_pred, original_pred.shape)
        original_pred = torch.nn.functional.softmax(original_pred, dim=0)
        label_conf, pred_label = imagenet_backselect.get_k_highest_val(original_pred, label_rank)
        is_correct = bool(pred_label == true_label)
        print('Original confidence: ', label_conf)
        print('Original pred label: ', pred_label, dataset.classes[pred_label])
        print('Is correct: ', is_correct)
        print('True Class: ', dataset.classes[true_label])
        if not is_correct or label_conf < 0.9:
            print('skipping\n\n')
            continue

        bs_result = batched_gradient_sis_results[i]

        # Show mask order.
        plt.figure(figsize=(3,3))
        plt.imshow(bs_result.mask_order.sum(axis=0))
        plt.xticks([], [])
        plt.yticks([], [])
        plt.show()

        # Plot backselect curve.
        plt.plot(np.arange(len(bs_result.confidences_over_backselect)),
                 bs_result.confidences_over_backselect[:, original_pred.argmax()])
        plt.axhline(y=0.9, c='black', linestyle='--')
        plt.show()

        # Find SIS mask for confidence >= threshold.
        mask = find_sis_mask_from_backselect_result(bs_result, SIS_THRESHOLD)
        fully_masked_image = torch.zeros(original_image.shape)
        masked_image = torch.where(mask, original_image, fully_masked_image)
        mask_size = int(mask.sum())
        mask_size_frac = mask_size / float(
            original_image.shape[1] * original_image.shape[2])
        print('Num pixels unmasked:  %mask_after_iters (%.1f%%)' % (mask_size, mask_size_frac * 100.))

        # Print masked image.
        all_red_image = torch.zeros(original_image.shape)
        all_red_image[0, :, :] = original_image.max()
        all_red_image[1, :, :] = original_image.min()
        all_red_image[2, :, :] = original_image.min()

        plt.figure(figsize=(4, 10))
        grid = utils.make_grid(
            torch.stack(
                [original_image,
                 torch.where(~mask, original_image, all_red_image),
                 masked_image]),
            nrow=1,
            padding=7,
            normalize=True,
        )
        show(grid)
        plt.show()

        # Confidence on masked image.
        preds = model(masked_image.unsqueeze(0).to(DEVICE))
        confidences = torch.nn.functional.softmax(preds, dim=1).detach().cpu().numpy()
        print('Predicted confidence on masked image: ',
              confidences.max(), confidences.argmax())
        print('\n\n')

if __name__ == '__main__':
    resnet_try()