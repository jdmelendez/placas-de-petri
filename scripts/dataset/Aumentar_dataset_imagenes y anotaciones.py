"""
    SE TIENE QUE EJECUTAR ESTE SCRIPT EN LA CARPETA QUE CONTIENE LA CARPETA DE ANOTACIONES E IMAGENES
"""

import cv2
import os
import glob
import io
import numpy as np
import pandas as pd
import xml.etree.cElementTree as ET
import shutil

# Obtenemos la ruta desde donde se ejecuta nuestro script
dir_actual = "C:/Users/jmelendez/GIT/placas-de-petri/laboratory"
# dir_actual = "C:/Users/jmelendez/GIT/placas-de-petri/laboratory"

# ______________________________________________________________________________________________________________________
# FUNCION ROTACIÓN:


def rotacion_imagen(imagen, angulo, centro=None, escala=1.0):
    # Obtenemos ancho y alto
    ancho = imagen.shape[1]
    alto = imagen.shape[0]

    if centro is None:
        centro = (ancho // 2, alto // 2)

    # Matriz de transformación, rotación
    M = cv2.getRotationMatrix2D(centro, angulo, escala)

    # Aplicamos rotación a la imagen
    ImagenRotada = cv2.warpAffine(imagen, M, (ancho, alto))

    # Devolvemos la imagen rotada
    return ImagenRotada


def rotacion_anotacion(boxes, imagen, angulo, centro=None):
    new_boxes = []
    ancho = imagen.shape[1]
    alto = imagen.shape[0]

    c, s = np.cos(np.deg2rad(-angulo)), np.sin(np.deg2rad(-angulo))
    R = np.array(((c, -s), (s, c)))

    if centro is None:
        rc = np.array(((ancho // 2, alto // 2),)).T

    for j in boxes:
        xmin = int(j[0])
        ymin = int(j[1])
        width = int(j[2])
        heigth = int(j[3])

        x1 = xmin
        y1 = ymin
        x2 = xmin
        y2 = ymin + heigth
        x3 = xmin + width
        y3 = ymin + heigth
        x4 = xmin + width
        y4 = ymin

        pts = np.array(((x1, y1), (x2, y2), (x3, y3), (x4, y4))).T
        pts_new = rc + R @ (pts - rc)

        xmin_new = int(min(pts_new[0]))
        ymin_new = int(min(pts_new[1]))
        heigth_new = width
        width_new = heigth

        if angulo == 180:
            heigth_new = heigth
            width_new = width

        new_boxes.append([xmin_new, ymin_new, width_new, heigth_new])
    return new_boxes
# ______________________________________________________________________________________________________________________
# FUNCION INVERTIR


def invertir_imagen(imagen):
    ImagenInvertida = cv2.flip(imagen, 1)

    return ImagenInvertida


def invertir_anotacion(boxes, imagen, angulo, centro=None):
    new_boxes = []
    ancho = imagen.shape[1]
    alto = imagen.shape[0]

    c, s = np.cos(np.deg2rad(-angulo)), np.sin(np.deg2rad(-angulo))
    R = np.array(((c, -s), (s, c)))

    if centro is None:
        rc = np.array(((ancho // 2, alto // 2),)).T

    for j in boxes:
        xmin = int(j[0])
        ymin = int(j[1])
        width = int(j[2])
        heigth = int(j[3])

        x1 = xmin
        y1 = ymin
        x2 = xmin
        y2 = ymin + heigth
        x3 = xmin + width
        y3 = ymin + heigth
        x4 = xmin + width
        y4 = ymin

        pts = np.array(((x1, y1), (x2, y2), (x3, y3), (x4, y4))).T
        pts_new = rc + R @ (pts - rc)

        xmin_new = -xmin + ancho - width
        ymin_new = ymin
        heigth_new = heigth
        width_new = width

        new_boxes.append([xmin_new, ymin_new, width_new, heigth_new])
    return new_boxes

# ______________________________________________________________________________________________________________________
# FUNCION NUEVAS ANOTACIONES EN XML:


def new_anottation(anotaciones, box_new, Nombre_Anotacion):
    root = anotaciones.getroot()
    for j in root:

        # if (len(root) == 1):
        #     j = root[0]
        # else:
        #     if ((root[0]).text == "- ") and ((root[1]).text != "- "):
        #         j = root[1]
        #     else:
        #         j = root[0]
        data_string_old = j.text
        if data_string_old != '- ':
            break

    data_string_new = ""
    for i in box_new:
        c = 0
        for j in i:
            if c == 0:
                data_string_new += f"{j},"
            elif c == 1:
                data_string_new += f"{j},"
            elif c == 2:
                data_string_new += f"{j},"
            elif c == 3:
                data_string_new += f"{j}; "
            c += 1

    escribe_en_clase1 = 1

    for i in root:
        if (len(root) == 1):
            i.text = i.text.replace(data_string_old, data_string_new)
        else:
            if escribe_en_clase1:
                i.text = i.text.replace(data_string_old, data_string_new)
                escribe_en_clase1 = 0
            else:
                i.text = '- '
                escribe_en_clase1 = 1

    anotaciones.write(Nombre_Anotacion)

# _____________________________________________________________________________________________________________________
# LECTURA DE IMÁGENES EN CARPETA


# Buscamos las carpetas de imagenes y anotaciones
carpetas = os.listdir(dir_actual)

# El archivo de la lista de carpetas ".py", lo quitamos de la lista
carpetas = [x for x in carpetas if "." not in x]
print("\n\nCARPETAS EXISTENTES:")
print(carpetas)

# Obtenemos las rutas de las distintas carpetas
paths = []
for i in range(len(carpetas)):
    paths.append(os.path.join(dir_actual, carpetas[i]))

# Vemos la cantidad de archivos de cada clase
print("\nCANTIDAD DE ARCHIVOS EN CADA CARPETA:")

cantidad_archivos = []
for x in range(len(carpetas)):
    cantidad_archivos.append(os.listdir(paths[x]))
    print(f"\tCarpeta '{carpetas[x]}': {len(cantidad_archivos[x])} archivos")


if len(cantidad_archivos[0]) == len(cantidad_archivos[1]):
    print("Misma cantidad de archivos en cada carpeta --> CONTINUE !")
else:
    print("Distinta cantidad de archivos en cada carpeta --> STOP !")
    quit()


# ______________________________________________________________________________________________________________________
# LECTURA DE FICHEROS EN CARPETA

# Imagenes:
imagenes = [cv2.imread(file) for file in glob.glob(f"{paths[1]}/*.png")]
nombres_imagenes = os.listdir(paths[1])

# Anotaciones:
anotaciones = [ET.parse(file)for file in glob.glob(f"{paths[0]}/*.xml")]
nombres_anotaciones = os.listdir(paths[0])

# ______________________________________________________________________________________________________________________
# OBTENCION DE ANOTACIONES ORIGINALES

roots = []
data_anotaciones = []
data_anotaciones_vacias = []
nombres_anotaciones_vacias = []
nombres_anotaciones_llenas = []
nombres_imagenes_llenas = []
nombres_imagenes_vacias = []
anotaciones_llenas = []

data = []
df = []
header = ["xmin", "ymin", "width", "heigth"]
boxes = []
all_boxes = []
existen_anotaciones_llenas = 0

for i in range(len(anotaciones)):

    roots.append(anotaciones[i].getroot())

    if (len(roots[i])) == 1:
        j = roots[i][0]
    else:
        if ((roots[i][0]).text == "- ") and ((roots[i][1]).text != "- "):
            j = roots[i][1]
        else:
            j = roots[i][0]

    data_string = j.text
    if data_string != "- ":
        data_anotaciones.append(j.text)
        nombres_anotaciones_llenas.append(nombres_anotaciones[i])
        nombres_imagenes_llenas.append(nombres_imagenes[i])
        anotaciones_llenas.append(anotaciones[i])
        existen_anotaciones_llenas = 1

    else:
        data_anotaciones_vacias.append(j.text)
        nombres_anotaciones_vacias.append(nombres_anotaciones[i])
        nombres_imagenes_vacias.append(nombres_imagenes[i])


if existen_anotaciones_llenas == 1:
    for i in range(len(data_anotaciones)):
        boxes = []
        data.append(io.StringIO(data_anotaciones[i].strip()))
        df.append(pd.read_csv(
            data[i], sep=",", header=None, names=header, lineterminator=";"))
        for index, row in df[i].iterrows():
            xmin = int(row['xmin'])
            ymin = int(row['ymin'])
            width = int(row['width'])
            heigth = int(row['heigth'])
            boxes.append([xmin, ymin, width, heigth])
        all_boxes.append(boxes)


# ______________________________________________________________________________________________________________________
# APLICAMOS LAS TRANSFORMACIONES:

# Recorremos cada carpeta y aplicamos la transformacion sobre cada imagen y cada coordenada de all_boxes:
# all boxes contiene: [boxes_imagen1 , boxes_imagen2,...] , a su vez, boxes_imagen1 contiene_ [[1,24,5,35],[5,67,46,5]...]
#

print(f"\n\nTransformando anotaciones...")

for y in range(len(nombres_anotaciones_llenas)):
    nombre_anotacion = nombres_anotaciones_llenas[y]

    AnotacionRotada90 = rotacion_anotacion(all_boxes[y], imagenes[y], 90)
    AnotacionRotada180 = rotacion_anotacion(all_boxes[y], imagenes[y], 180)
    AnotacionRotada270 = rotacion_anotacion(all_boxes[y], imagenes[y], 270)
    AnotacionInvertida = invertir_anotacion(all_boxes[y], imagenes[y], 270)
    AnotacionRotada90Inv = rotacion_anotacion(
        AnotacionInvertida, imagenes[y], 90)
    AnotacionRotada180Inv = rotacion_anotacion(
        AnotacionInvertida, imagenes[y], 180)
    AnotacionRotada270Inv = rotacion_anotacion(
        AnotacionInvertida, imagenes[y], 270)
    new_anottation(anotaciones_llenas[y], AnotacionRotada90,
                   f"C:/Users/jmelendez/GIT/placas-de-petri/laboratory/Annotations/r90_{nombre_anotacion}")
    new_anottation(anotaciones_llenas[y], AnotacionRotada180,
                   f"C:/Users/jmelendez/GIT/placas-de-petri/laboratory/Annotations/r180_{nombre_anotacion}")
    new_anottation(anotaciones_llenas[y], AnotacionRotada270,
                   f"C:/Users/jmelendez/GIT/placas-de-petri/laboratory/Annotations/r270_{nombre_anotacion}")
    new_anottation(anotaciones_llenas[y], AnotacionInvertida,
                   f"C:/Users/jmelendez/GIT/placas-de-petri/laboratory/Annotations/Inv_{nombre_anotacion}")
    new_anottation(anotaciones_llenas[y], AnotacionRotada90Inv,
                   f"C:/Users/jmelendez/GIT/placas-de-petri/laboratory/Annotations/Inv_r90_{nombre_anotacion}")
    new_anottation(anotaciones_llenas[y], AnotacionRotada180Inv,
                   f"C:/Users/jmelendez/GIT/placas-de-petri/laboratory/Annotations/Inv_r180_{nombre_anotacion}")
    new_anottation(anotaciones_llenas[y], AnotacionRotada270Inv,
                   f"C:/Users/jmelendez/GIT/placas-de-petri/laboratory/Annotations/Inv_r270_{nombre_anotacion}")

for y in range(len(nombres_anotaciones_vacias)):
    nombre_anotacion = nombres_anotaciones_vacias[y]

    shutil.copy(f"C:/Users/jmelendez/GIT/placas-de-petri/laboratory/Annotations/{nombre_anotacion}",
                f"C:/Users/jmelendez/GIT/placas-de-petri/laboratory/Annotations/r90_{nombre_anotacion}")
    shutil.copy(f"C:/Users/jmelendez/GIT/placas-de-petri/laboratory/Annotations/{nombre_anotacion}",
                f"C:/Users/jmelendez/GIT/placas-de-petri/laboratory/Annotations/r180_{nombre_anotacion}")
    shutil.copy(f"C:/Users/jmelendez/GIT/placas-de-petri/laboratory/Annotations/{nombre_anotacion}",
                f"C:/Users/jmelendez/GIT/placas-de-petri/laboratory/Annotations/r270_{nombre_anotacion}")
    shutil.copy(f"C:/Users/jmelendez/GIT/placas-de-petri/laboratory/Annotations/{nombre_anotacion}",
                f"C:/Users/jmelendez/GIT/placas-de-petri/laboratory/Annotations/Inv_{nombre_anotacion}")
    shutil.copy(f"C:/Users/jmelendez/GIT/placas-de-petri/laboratory/Annotations/{nombre_anotacion}",
                f"C:/Users/jmelendez/GIT/placas-de-petri/laboratory/Annotations/Inv_r90_{nombre_anotacion}",)
    shutil.copy(f"C:/Users/jmelendez/GIT/placas-de-petri/laboratory/Annotations/{nombre_anotacion}",
                f"C:/Users/jmelendez/GIT/placas-de-petri/laboratory/Annotations/Inv_r180_{nombre_anotacion}")
    shutil.copy(f"C:/Users/jmelendez/GIT/placas-de-petri/laboratory/Annotations/{nombre_anotacion}",
                f"C:/Users/jmelendez/GIT/placas-de-petri/laboratory/Annotations/Inv_r270_{nombre_anotacion}")

print(f"\n\nTransformando imagenes...")

for y in range(len(imagenes)):
    nombre_imagen = nombres_imagenes[y]

    ImagenRotada90 = rotacion_imagen(imagenes[y], 90, None, 1.0)
    ImagenRotada180 = rotacion_imagen(imagenes[y], 180, None, 1.0)
    ImagenRotada270 = rotacion_imagen(imagenes[y], 270, None, 1.0)
    ImagenInvertida = invertir_imagen(imagenes[y])
    ImagenRotada90Inv = rotacion_imagen(ImagenInvertida, 90, None, 1.0)
    ImagenRotada180Inv = rotacion_imagen(ImagenInvertida, 180, None, 1.0)
    ImagenRotada270Inv = rotacion_imagen(ImagenInvertida, 270, None, 1.0)
    cv2.imwrite(
        f"C:/Users/jmelendez/GIT/placas-de-petri/laboratory/Images/r90_{nombre_imagen}", ImagenRotada90)
    cv2.imwrite(
        f"C:/Users/jmelendez/GIT/placas-de-petri/laboratory/Images/r180_{nombre_imagen}", ImagenRotada180)
    cv2.imwrite(
        f"C:/Users/jmelendez/GIT/placas-de-petri/laboratory/Images/r270_{nombre_imagen}", ImagenRotada270)
    cv2.imwrite(
        f"C:/Users/jmelendez/GIT/placas-de-petri/laboratory/Images/Inv_{nombre_imagen}", ImagenInvertida)
    cv2.imwrite(
        f"C:/Users/jmelendez/GIT/placas-de-petri/laboratory/Images/Inv_r90_{nombre_imagen}", ImagenRotada90Inv)
    cv2.imwrite(
        f"C:/Users/jmelendez/GIT/placas-de-petri/laboratory/Images/Inv_r180_{nombre_imagen}", ImagenRotada180Inv)
    cv2.imwrite(
        f"C:/Users/jmelendez/GIT/placas-de-petri/laboratory/Images/Inv_r270_{nombre_imagen}", ImagenRotada270Inv)


print(f"\n\nNuevas imagenes y anotaciones creadas!")
