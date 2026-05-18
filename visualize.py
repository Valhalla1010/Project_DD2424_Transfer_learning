import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import numpy as np
import torchvision
from torch.utils.data import DataLoader
from BinaryClass import Transforms, BinaryPetDataset

def imshow(img, title=None):
    img = img.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)
    plt.figure(figsize=(8, 4))
    plt.imshow(img)
    plt.axis("off")
    if title is not None:
        plt.title(title)
    plt.show(block=True)


def visualize_batch(dataloader):
    images, labels = next(iter(dataloader))
    images = images.cpu()
    labels = labels.cpu()
    grid = torchvision.utils.make_grid(images[:8], nrow=4)
    title = " | ".join(
        ["cat" if int(x.item()) == 0 else "dog" for x in labels[:8]]
    )
    imshow(grid, title)


def main():
    root = "Datasets"
    train_trasform, test_transform = Transforms()
    # load data
    dataset = BinaryPetDataset(root,  "trainval.txt", train_trasform)
    loader = DataLoader(dataset, batch_size=8)
    visualize_batch(loader)


if __name__ == "__main__":
    main()