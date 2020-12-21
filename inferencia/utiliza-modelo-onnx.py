import io
import numpy as np
import torch.onnx
import onnx
import onnxruntime
import torch
import torchvision
import glob
from torchvision import transforms
from PIL import Image
from onnx import optimizer
import time
# from onnxruntime_tools import optimizer


path_modelo_deteccion = "./models_trained/544-CH-CA.onnx"
path_modelo_clasificacion = "./models_trained/797-AG-BC.onnx"

path_imagenes_deteccion = "./inferencia/imgs_prueba"
path_imagenes_clasificacion = "./inferencia/imgs_prueba"

FLAG_DETECCION = 1
FLAG_CLASIFICACION = 0


def tr_deteccion(imagen):
    imagen = Image.open(imagen)
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])

    loader = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(std, mean)])
    # imagen = imagen.numpy()
    imagen = loader(imagen).float()
    imagen = imagen.numpy()

    return imagen


def tr_clasificacion(imagen):

    imagen = Image.open(imagen)
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])

    loader = transforms.Compose(
        [transforms.Resize(128), transforms.ToTensor(), transforms.Normalize(std, mean)])

    imagen = loader(imagen).float()
    imagen = imagen.unsqueeze(0)

    return imagen


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


if FLAG_DETECCION:

    ort_deteccion = onnxruntime.InferenceSession(path_modelo_deteccion)

    # Lee imagenes e inferencia
    paths_imagenes = glob.glob(f"{path_imagenes_deteccion}/*.png")

    imagenes = [tr_deteccion(path_imagen) for path_imagen in paths_imagenes]

    for imagen in imagenes:

        time_start = time.time()
        salidas = ort_deteccion.run(None, {"image.1": imagen})
        # print(salidas)
        time_end = time.time()
        total_time = time_end - time_start
        prediccion = salidas
        print(f"Tiempo de inferencia deteccion:{total_time:.2f} sec")


if FLAG_CLASIFICACION:
    ort_clasificacion = onnxruntime.InferenceSession(path_modelo_clasificacion)

    # Lee imagenes e inferencia
    paths_imagenes = glob.glob(f"{path_imagenes_clasificacion}/*.png")

    imagenes = [tr_clasificacion(path_imagen)
                for path_imagen in paths_imagenes]

    for imagen in imagenes:

        entradas = {ort_clasificacion.get_inputs()[0].name: to_numpy(imagen)}
        salidas = ort_clasificacion.run(None, entradas)
        prediccion = salidas[0]
        print(prediccion)
