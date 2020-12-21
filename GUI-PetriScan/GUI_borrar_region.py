from PyQt5.QtGui import QColor, QImage, QPixmap, QPen, QPainter, QFont


def borrar_region(painter, coordenadas_click, path_archivo, color, px, offset_X=0, offset_Y=0, factor_escala_x=0, factor_escala_y=0):

    path_fichero_txt = path_archivo[:-3] + 'txt'

    boxes_existentes = []

    with open(f'{path_fichero_txt}') as texto:
        for indice, linea in enumerate(texto):
            if indice == 0:
                continue

            linea = linea.split(" ", 3)
            boxes_existentes.append(
                [int(linea[0]), int(linea[1]), int(linea[2]), int(linea[3])])

    box_a_borrar, click_correcto = obten_region_a_borrar(
        coordenadas_click, boxes_existentes, factor_escala_x, factor_escala_y, offset_X, offset_Y)

    if click_correcto:
        pen = QPen()
        pen.setColor(QColor(color))
        pen.setWidth(px)
        painter.setPen(pen)
        painter.drawRect((box_a_borrar[0]*factor_escala_x), (box_a_borrar[1]*factor_escala_y),
                         (box_a_borrar[2]-box_a_borrar[0])*factor_escala_x, (box_a_borrar[3]-box_a_borrar[1])*factor_escala_y)

    return box_a_borrar, click_correcto


def obten_region_a_borrar(coordenadas_click, boxes_existenes, factor_escala_x, factor_escala_y, offset_X, offset_Y):

    x = (coordenadas_click[0]-offset_X) / factor_escala_x
    y = (coordenadas_click[1]-offset_Y) / factor_escala_y

    lista_propuesta_regiones_a_borrar = []
    lista_areas_propuestas_regiones_a_borrar = []

    # Obtenemos las regiones cuyo click esta contenido en su interior
    for box in boxes_existenes:
        xmax_box = int(box[2])
        ymax_box = int(box[3])
        xmin_box = int(box[0])
        ymin_box = int(box[1])

        if x > xmin_box and y > ymin_box and x < xmax_box and y < ymax_box:
            propuesta_region_a_borar = [xmin_box, ymin_box, xmax_box, ymax_box]
            lista_areas_propuestas_regiones_a_borrar.append(
                calcula_area_region(xmin_box, ymin_box, xmax_box, ymax_box))
            lista_propuesta_regiones_a_borrar.append(propuesta_region_a_borar)

    # En caso de clicar en una zona de union entre dos regiones, decidimos por la de area mas pequeña
    if len(lista_propuesta_regiones_a_borrar) > 1:
        indice_area_menor = lista_areas_propuestas_regiones_a_borrar.index(
            min(lista_areas_propuestas_regiones_a_borrar))
        region_a_borrar = lista_propuesta_regiones_a_borrar[indice_area_menor]
        click_correcto = True

    elif len(lista_propuesta_regiones_a_borrar) == 1:
        region_a_borrar = lista_propuesta_regiones_a_borrar[0]
        click_correcto = True
    else:
        region_a_borrar = []
        click_correcto = False

    return region_a_borrar, click_correcto


def calcula_area_region(xmin, ymin, xmax, ymax):
    area = (xmax - xmin) * (ymax - ymin)
    return area
