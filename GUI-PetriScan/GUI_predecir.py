import torch
from GUI_filtro import filtro_boxes
import numpy as np
from GUI_config import CLASES_CLASIFICACION
import time


def predecir(modelos, imagen, device, indice, id_imagenes_lista):
    """Esta funcion se encarga de realizar la inferencia de una unica imagen. Para ello, se le han de pasar los modelos
    disponibles, y en funcion del identificador de la imagen (asociado a su nombre), se elige un modelo u otro. Además,
    se le ha de pasar el dispositivo en el que debe realizarse, bien sea la CPU o GPU. Se encarga de ver que tipo de 
    inferencia se debe realizar, si clasifcacion o deteccion. 

    Args:
        modelos ([type]): Diccionario de modelos  
        imagen ([type]): Matriz con los valores pixelicos de la imagen ya tranformada en tensor
        device ([type]): Dispositivo en el cual se debe de realizar la inferencia. CUDA o CPU.
        indice ([type]): Imagen que toca inferenciar en la lista de imagenes.
        id_imagenes_lista ([type]): Lista con los ids de cada imagen para poder cargar el modelo asociado.

    Returns:
        [type]: Se devuelve la prediccion, bien sea de deteccion (coordenadas de las regiones) o clasificacion (clase)
    """

    modelo, tecnica = modelos[id_imagenes_lista[indice]]

    if tecnica == "DETECCION":
        prediccion_sin_filtro, scores_sin_filtro = predecir_deteccion(
            modelo, imagen, device)
        prediccion = filtro_boxes(prediccion_sin_filtro, scores_sin_filtro)

    else:
        prediccion = predecir_clasificacion(modelo, imagen, device)

    return prediccion


def predecir_deteccion(modelo, imagen, device):
    """A traves de esta funcion se predicen las coordenadas de las colonias. Simplemente, se le pasa el modelo seleccionado,
    la imagen en cuestion, y el dispositivo donde se ejecuta. 

    Args:
        modelo ([type]): Modelo seleccionado para la inferencia segun el identificador de imagen
        imagen ([type]): Tensor que contiene los valores pixelicos de la imagen
        device ([type]): Dispositivo donde se ejecuta la inferencia. CUDA o CPU.

    Returns:
        [type]: Devuelve las coordenadas de las regiones y los niveles de confianza para cada una de ellas. 
    """

    with torch.no_grad():

        imagen = imagen.to(device)
        start_time = time.time()
        prediction = modelo([imagen])
        end_time = time.time()
        print(f"Tiempo ejecucion deteccion:{(end_time - start_time):.2f} sec")

        # Trabajamos las predicciones
        boxes = prediction[0]["boxes"]
        scores = prediction[0]["scores"]
        boxes = boxes.cpu().numpy()
        scores = scores.cpu().numpy()

    return boxes, scores


def predecir_clasificacion(modelo, imagen, device):
    """A traves esta funcion se realiza la inferencia de la imagen en cuestion. Concretamente, se aborda la tarea de 
    clasificacion, detectando si la placa pertenece a la clase 'presencia' o a la clase 'ausencia'.

    Args:
        modelo ([type]): modelo seleccionado para realizar la inferencia
        imagen ([type]): tensor con las coordenadas pixelicas de la imagen
        device ([type]): dispositivo donde se ejecuta la infernecia, CUDa o CPU.

    Returns:
        [type]: se devuelve un string con la palabra 'ausencia' o 'presencia'.
    """

    imagen = imagen.to(device)
    start_time = time.time()
    prediction = modelo(imagen)
    end_time = time.time()
    print(f"Tiempo ejecucion clasificacion:{(end_time - start_time):.2f} sec")

    # Trabajamos las predicciones
    _, pred = torch.max(prediction, 1)
    pred = np.squeeze(pred.cpu().numpy())

    # Convertimos prediccion
    pred = CLASES_CLASIFICACION[pred]

    return pred
