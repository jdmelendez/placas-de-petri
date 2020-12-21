"""
Mediante este script se generan imagenes de colonias a partir de las contornos de cada colonia en la imagen.

Se ha de definir la ruta original de las Imagenes y las Anotaciones, y la ruta final donde se guardaran los recortes.
"""
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from operator import itemgetter
from heapq import nlargest
import xml.etree.cElementTree as ET
import copy
import pandas as pd
import io
import os
import sys


PATH_IMGS = './datasets/dataset-deteccion/ToDrive/train/Images'
PATH_ANNS = './datasets/dataset-deteccion/ToDrive/train/Annotations'
PATH_IMGS_CROP = './datasets/dataset-deteccion-colonias/Images'
PATH_ANNS_CROP = './datasets/dataset-deteccion-colonias/Annotations'



def obtener_paths_imagenes(path=PATH_IMGS):
    paths_imagenes = glob.glob(f"{path}/*.png")

    return paths_imagenes


def obtener_paths_anotaciones(path=PATH_ANNS):
    paths_anotaciones = glob.glob(f"{path}/*.xml")

    return paths_anotaciones


def obtener_tamaño_imagen(path_imagen):
    img = cv2.imread(path_imagen)
    alto, ancho, profundidad = img.shape

    return alto, ancho



def crop_imagen(path_imagen, alto_crop, ancho_crop, alto, ancho):

    ancho_ini = 0
    alto_ini = 0
    indice = 1
    img = cv2.imread(path_imagen)

    while ancho_ini < ancho:
        img_crop = img[alto_ini:alto, ancho_ini:ancho_crop*indice]
        ancho_ini += ancho_crop
        guarda_imagen_crop(path_imagen, indice, img_crop)
        indice += 1


def crop_imagen_por_contornos(path_imagen, contorno, path_anotacion, PATH_NUEVAS_IMGS=PATH_IMGS_CROP):

    img = cv2.imread(path_imagen)

    xmin = contorno[0]
    ymin = contorno[1]
    xmax = contorno[2]
    ymax = contorno[3]

    img_crop = img[ymin:ymax, xmin:xmax]

    nuevo_nombre_imagen = path_anotacion.split("\\")[-1][:-3]+'png'
    nuevo_path_imagen = os.path.join(PATH_NUEVAS_IMGS, nuevo_nombre_imagen)

    cv2.imwrite(nuevo_path_imagen, img_crop)


def guarda_imagen_crop(path_imagen, indice, img_crop, path_imagenes_crop=PATH_IMGS_CROP):
    nombre_imagen = path_imagen.split("\\")[-1]

    nombre_imagen = f"{nombre_imagen[:-4]}_{indice}.png"
    cv2.imwrite(f'{path_imagenes_crop}/{nombre_imagen}', img_crop)


def forma_nombre_anotacion(path_anotacion, indice_ampolla, PATH_NUEVAS_ANNS=PATH_ANNS_CROP):
    nombre_anotacion = path_anotacion.split("\\")[-1]

    nuevo_nombre_anotacion = nombre_anotacion[:-4] + \
        f'_{indice_ampolla}'+'.xml'

    nuevo_path_anotacion = os.path.join(
        PATH_NUEVAS_ANNS, nuevo_nombre_anotacion)

    return nuevo_path_anotacion


def obtener_contornos_colonias(path_anotacion, path_imagen):

    annotation_xml = ET.parse(path_anotacion)
    root = annotation_xml.getroot()
    header = ["xmin", "ymin", "width", "heigth"]
    boxes = []

    data_string = (root[0]).text

    if len(root) != 1:
        if (root[1]).text != '- ':
            data_string = (root[1]).text

    if data_string == "- ":
        os.remove(path_anotacion)
        os.remove(path_imagen)

    else:
        No_labels = False
        data = io.StringIO(data_string.strip())
        df = pd.read_csv(data, sep=",", header=None,
                         names=header, lineterminator=";")

        # Obtenemos boxes
        for index, row in df.iterrows():
            xmax = int((row['width']))+int((row['xmin']))
            ymax = int((row['heigth']))+int((row['ymin']))
            xmin = int((row['xmin']))
            ymin = int((row['ymin']))
            boxes.append([xmin, ymin, xmax, ymax])

    return boxes


paths_imagenes = obtener_paths_imagenes()
paths_anotaciones = obtener_paths_anotaciones()

for indice, path_imagen in enumerate(paths_imagenes):

    if DETECTAR_AMPOLLAS:
        path_anotacion = paths_anotaciones[indice]
        contornos_colonias = obtener_contornos_colonias(
            path_anotacion, path_imagen)

        for indice, contorno in enumerate(contornos_colonias):
            nuevo_path_anotacion = forma_nombre_anotacion(path_anotacion, indice)
            crop_imagen_por_contornos(path_imagen, contorno, nuevo_path_anotacion)
