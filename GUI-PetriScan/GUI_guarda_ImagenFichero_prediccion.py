import matplotlib.pyplot as plt
from matplotlib import patches
import cv2

# CALCULAR DPI SEGUN LA RESOLUCION DE LA PANTALLA EN ESTA PAGINA WEB : https://dpi.lv/


def guarda_ImagenFichero_prediccion(diccionario_imgs_preds):

    for path_imagen, prediccion in diccionario_imgs_preds.items():
        prediccion = prediccion[0]

        if isinstance(prediccion, str):
            guarda_resultados_clasificacion(
                prediccion=prediccion, path_imagen=path_imagen)
        else:
            guarda_resultados_deteccion(
                boxes=prediccion, path_imagen=path_imagen)


def guarda_resultados_deteccion(boxes, path_imagen):

    imagen = cv2.imread(path_imagen, cv2.IMREAD_COLOR)
    # Guardamos un txt con los datos de las regiones
    archivo = open(f"{path_imagen[:-4]}_OK.txt", "a")
    archivo.write("xmin ymin xmax ymax\n")

    text = f"Colonias: {len(boxes)}"
    font = cv2.FONT_HERSHEY_DUPLEX

    # cv2.putText(imagen, text, (15, 60), font, 1.5, (0, 0, 0), 2)

    for i in boxes:
        xmax = int(i[2])
        ymax = int(i[3])
        xmin = int(i[0])
        ymin = int(i[1])
        width = xmax - xmin
        heigth = ymax - ymin

        cv2.rectangle(imagen, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)

        datos_archivo = f"{xmin} {ymin} {xmax} {ymax}\n"

        archivo.write(datos_archivo)

    archivo.close()
    cv2.imwrite(f"{path_imagen[:-4]}_OK.png", imagen)


def guarda_resultados_clasificacion(prediccion, path_imagen):

    imagen = cv2.imread(path_imagen, cv2.IMREAD_COLOR)

    # Guardamos un txt con los datos de las regiones
    archivo = open(f"{path_imagen[:-4]}_OK.txt", "a")
    archivo.write(prediccion)
    archivo.close()

    text = f"Clase: {prediccion}"
    font = cv2.FONT_HERSHEY_DUPLEX
    # cv2.putText(imagen, text, (15, 60), font, 1.5, (0, 0, 0), 2)
    cv2.imwrite(f"{path_imagen[:-4]}_OK.png", imagen)
