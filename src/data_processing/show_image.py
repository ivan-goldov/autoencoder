import numpy as np
from matplotlib import pyplot as plt
from torch import Tensor


def show_image(img: Tensor):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
