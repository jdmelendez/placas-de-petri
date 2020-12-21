from torchvision.transforms import functional as T
import matplotlib.pyplot as plt
from torchvision import transforms
import numpy as np
from PIL import Image


def A(imagen):
    imagen = plt.imread(imagen)
    imagen = T.to_tensor(imagen)

    return imagen


def B(imagen):

    imagen = Image.open(imagen)
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])

    loader = transforms.Compose(
        [transforms.Resize(128), transforms.ToTensor(), transforms.Normalize(std, mean)])
    imagen = loader(imagen).float()
    imagen = imagen.unsqueeze(0)

    return imagen
