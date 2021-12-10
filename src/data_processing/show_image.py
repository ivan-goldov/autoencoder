import numpy as np
from matplotlib import pyplot as plt
from torch import Tensor


def show_image(img: Tensor):
    img = img.detach().cpu().numpy() / 2 + 0.5
    plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.show()
