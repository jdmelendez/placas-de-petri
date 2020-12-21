import math
import numpy as np


def filtro_boxes(boxes, scores, iou_threshold=0.2, dist_eucl_threshold=55, score_threshold=0.25):
    """Esta funcion realiza un filtrado de las anotaciones creadas por la prediccion de manera que se 
    eliminan aquellas que se solapan o superponen con otras en función de unos umbrales. Estos umbrales
    son el IOU (area de interseccion/area de union) y la distancia euclidea entre los centros de dos regiones.
    Seguidamente, se eliminan aquellas regiones que no superen el indice de confianza establecido en el 
    parametro score_threshold

    Args:
        boxes ([type]): Lista con las coordenadas de cada una de las regiones
        scores ([type]): Lista con los niveles de confianza de cada una de las regiones
        score_threshold ([type]): umbral de confianza
        iou_threshold (float, optional): umbral de IOU.
        dist_eucl_threshold (int, optional): umbral de distancia euclidea

    Returns:
        [type]: Se devuelve una lista con las coordenadas de las nuevas regiones filtradas.
    """
    boxes_filt = boxes
    scores_filt = scores

    indices_delete_boxes = []
    indice_i = 0

    for i in boxes:
        x1A = i[0]
        y1A = i[1]
        x2A = i[2]
        y2A = i[3]

        indice_j = 0
        for j in boxes[indice_i:]:
            iou = 0
            dist_eucl = 1000
            x1B = j[0]
            y1B = j[1]
            x2B = j[2]
            y2B = j[3]

            # Arriba izquierda - Abajo derecha
            if ((x1A < x1B) and (y1A < y1B) and (x2A < x2B) and (y2A < y2B) and (x1B < x2A) and (y1B < y2A)) or (
                    (x1A > x1B) and (y1A > y1B) and (x2A > x2B) and (y2A > y2B) and (x1A < x2B) and (y1A < y2B)):
                x_left = max(x1A, x1B)
                y_top = max(y1A, y1B)
                x_right = min(x2A, x2B)
                y_bottom = min(y2A, y2B)

            # Arriba derecha
            elif ((x1A < x1B) and (y1A > y1B) and (x2A < x2B) and (y2A > y2B) and (x1B < x2A) and (y1A < y2B)):
                x_left = x1B
                y_top = y1A
                x_right = x2A
                y_bottom = y2B

            # Abajo izquierda
            elif ((x1A > x1B) and (y1A < y1B) and (x2A > x2B) and (y2A < y2B) and (x1A < x2B) and (y1B < y2A)):
                x_left = x1A
                y_top = y1B
                x_right = x2B
                y_bottom = y2A

            # Arriba
            elif ((x1A < x1B) and (y1A > y1B) and (x2A > x2B) and (y2A > y2B) and (y1A < y2B) and (x1B < x2A)):
                x_left = x1B
                y_top = y1A
                x_right = x2B
                y_bottom = y2B

            # Derecha
            elif ((x1A < x1B) and (y1A < y1B) and (x2A < x2B) and (y2A > y2B) and (y1A < y2B) and (x1B < x2A)):
                x_left = x1B
                y_top = y1B
                x_right = x2A
                y_bottom = y2B

            # Abajo
            elif ((x1A < x1B) and (y1A < y1B) and (x2A > x2B) and (y2A < y2B) and (y1A < y2B) and (x1B < x2A)):
                x_left = x1B
                y_top = y1B
                x_right = x2B
                y_bottom = y2A

            # Izquierda
            elif ((x1A > x1B) and (y1A < y1B) and (x2A > x2B) and (y2A > y2B) and (y1A < y2B) and (x1B < x2A)):
                x_left = x1A
                y_top = y1B
                x_right = x2B
                y_bottom = y2B

            # Arriba ancho
            elif ((x1A > x1B) and (y1A > y1B) and (x2A < x2B) and (y2A > y2B) and (y1A > y2B) and (x1A < x2B)):
                x_left = x1A
                y_top = y1A
                x_right = x2A
                y_bottom = y2B

            # Bajo ancho
            elif ((x1A > x1B) and (y1A < y1B) and (x2A < x2B) and (y2A < y2B) and (y1A < y2B) and (x1A < x2B)):
                x_left = x1A
                y_top = y1B
                x_right = x2A
                y_bottom = y2A

            # Izquierda ancho
            elif ((x1A > x1B) and (y1A > y1B) and (x2A > x2B) and (y2A < y2B) and (y1A < y2B) and (x1A < x2B)):
                x_left = x1A
                y_top = y1A
                x_right = x2B
                y_bottom = y2A

            # Derecha ancho
            elif ((x1A < x1B) and (y1A > y1B) and (x2A < x2B) and (y2A < y2B) and (y1A < y2B) and (x1A < x2B)):
                x_left = x1B
                y_top = y1A
                x_right = x2A
                y_bottom = y2A

            # Horizontal
            elif ((x1A > x1B) and (y1A < y1B) and (x2A < x2B) and (y2A > y2B) and (y1A < y2B) and (x1A < x2B)):
                x_left = x1A
                y_top = y1B
                x_right = x2A
                y_bottom = y2B

            # Vertical
            elif ((x1A < x1B) and (y1A > y1B) and (x2A > x2B) and (y2A < y2B) and (y1A < y2B) and (x1A < x2B)):
                x_left = x1B
                y_top = y1A
                x_right = x2B
                y_bottom = y2A

            # Dentro
            elif (x1A < x1B and y1A < y1B and x2A > x2B and y2A > y2B) or (
                    (x1A > x1B and y1A > y1B and x2A < x2B and y2A < y2B)):
                centroAx = x1A + (x2A - x1A) / 2
                centroAy = y1A + (y2A - y1A) / 2
                centroBx = x1B + (x2B - x1B) / 2
                centroBy = y1B + (y2B - y1B) / 2
                dist_eucl = math.sqrt(
                    (centroAx - centroBx) ** 2 + (centroAy - centroBy) ** 2)

                if dist_eucl < dist_eucl_threshold:
                    boxes_filt[indice_i + indice_j][:] = [0, 0, 0, 0]
                indice_j += 1
                continue

            else:
                indice_j += 1
                continue

            # The intersection of two axis-aligned bounding boxes
            intersection_area = (x_right - x_left) * (y_bottom - y_top)

            # compute the area of both AABBs
            bb1_area = (x2A - x1A) * (y2A - y1A)
            bb2_area = (x2B - x1B) * (y2B - y1B)

            # compute the intersection over union
            iou = intersection_area / \
                float(bb1_area + bb2_area - intersection_area)

            if (iou > iou_threshold):
                boxes_filt[indice_i + indice_j][:] = [0, 0, 0, 0]

            indice_j += 1
        indice_i += 1

    for i in range(len(boxes_filt)):
        result = np.all((boxes_filt[i] == 0))
        if result:
            indices_delete_boxes.append(i)

    boxes_filt = np.delete(boxes_filt, indices_delete_boxes, axis=0)
    scores_filt = np.delete(scores_filt, indices_delete_boxes, axis=0)

    indices_delete_scores = []

    for indice, score in enumerate(scores_filt):
        if score < score_threshold:
            indices_delete_scores.append(indice)

    boxes_filt = np.delete(boxes_filt, indices_delete_scores, axis=0)
    # scores_filt = np.delete(scores_filt, indices_delete_scores, axis=0)

    return boxes_filt
