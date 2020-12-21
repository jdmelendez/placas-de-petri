import io
import numpy as np
import torch.onnx
#import onnx
import onnxruntime
import torch
import torchvision
import glob
from torchvision import transforms
from PIL import Image
import time


path_modelo_deteccion = "./models_trained/544-CH-CA.onnx"
path_modelo_clasificacion = "./models_trained/797-AG-BC.onnx"

path_imagenes_deteccion = "./imgs_prueba_deteccion"
path_imagenes_clasificacion = "./imgs_prueba_clasificacion"

FLAG_DETECCION = 1
FLAG_CLASIFICACION = 1


def tr_deteccion(imagen):
    imagen = Image.open(imagen)
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])

    loader = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(std, mean)])
    imagen = loader(imagen).float()

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
    # Inicia sesion runtime
    #modelo_onnx = onnx.load(path_modelo_deteccion)
    ort_deteccion = onnxruntime.InferenceSession(path_modelo_deteccion)

    # Lee imagenes e inferencia
    paths_imagenes = glob.glob(f"{path_imagenes_deteccion}/*.png")

    imagenes = [tr_deteccion(path_imagen) for path_imagen in paths_imagenes]

    for imagen in imagenes:

        imagen1 = imagen.numpy()
        time_start = time.time()
        salidas = ort_deteccion.run(None, {"image.1": imagen1})
        time_end = time.time()
        total_time = time_end - time_start
        prediccion = salidas
        print(f"Tiempo de inferencia deteccion:{total_time:.2f} sec")
        #print(prediccion)


if FLAG_CLASIFICACION:
    #modelo_onnx = onnx.load(path_modelo_clasificacion)
    ort_clasificacion = onnxruntime.InferenceSession(path_modelo_clasificacion)

    # Lee imagenes e inferencia
    paths_imagenes = glob.glob(f"{path_imagenes_clasificacion}/*.png")

    imagenes = [tr_clasificacion(path_imagen)
                for path_imagen in paths_imagenes]

    for imagen in imagenes:
        time_start = time.time()
        entradas = {ort_clasificacion.get_inputs()[0].name: to_numpy(imagen)}
        salidas = ort_clasificacion.run(None, entradas)
        time_end = time.time()
        total_time = time_end - time_start
        prediccion = salidas[0]
        print(f"Tiempo de inferencia clasificacion:{total_time:.2f} sec")
        #print(prediccion)
