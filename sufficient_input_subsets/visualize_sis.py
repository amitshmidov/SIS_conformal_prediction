import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets

from util import data_util


def visualize_sis(dataset, image_idx, sis_path="saved_models/resnet18_rep1/sis/cifar10_test"):
    image_idx = 30
    mask = np.load(f"{sis_path}/cifar10_test_{image_idx}.npz")["mask"]
    im = dataset[image_idx][0].permute((1, 2, 0))
    im -= im.min()
    im /= im.max()

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(im)
    plt.subplot(1, 2, 2)
    plt.imshow(im * mask.swapaxes(0, 2))
    plt.show()


if __name__ == '__main__':
    transform = data_util.cifar_test_transform()
    dataset = datasets.CIFAR10(root='data/',
                               train=False,
                               transform=transform,
                               download=True)
    visualize_sis(dataset=dataset, image_idx=0)
