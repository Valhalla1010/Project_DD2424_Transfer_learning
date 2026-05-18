import matplotlib.pyplot as plt
import numpy as np
import torchvision


def imshow(inp, title=None):
    #Show tensor image after denormalization.
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.axis("off")
    plt.show()

def visualize_batch(dataloader):
    # get one batch
    images, labels = next(iter(dataloader))
    # make grid
    grid = torchvision.utils.make_grid(images[:5])  # show first 16 images
    # convert labels to text
    title = [("cat" if x == 0 else "dog") for x in labels[:5]]
    imshow(grid, title=" | ".join(title))

# NAG optimizer
    #optimizer = optim.SGD(model.fc.parameters(), lr= 0.001, momentum=0.9, nesterov=True)

