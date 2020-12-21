import cv2


def valida_cambios_edicion(nuevo_resultado, path_archivo, flag_dibujar, flag_borrar, flag_edicion_clasificacion):

    path_fichero_txt = path_archivo[:-3] + 'txt'

    if flag_dibujar:
        # Sobreescribir imagen
        imagen = cv2.imread(path_archivo, cv2.IMREAD_COLOR)

        # Sobreescribir fichero txt
        fichero_txt = open(f'{path_fichero_txt}', 'a')

        for box in nuevo_resultado:
            xmax = int(box[2])
            ymax = int(box[3])
            xmin = int(box[0])
            ymin = int(box[1])

            datos_archivo = f"{xmin} {ymin} {xmax} {ymax}\n"
            fichero_txt.write(datos_archivo)

            cv2.rectangle(imagen, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)

        fichero_txt.close
        cv2.imwrite(f"{path_archivo}", imagen)

    elif flag_borrar:

        # Se lee el archivo de anotaciones y se elimina de el las regiones que han sido clickadas
        fichero_txt = open(f'{path_fichero_txt}', 'r')
        lineas = fichero_txt.readlines()
        fichero_txt.close()

        lista_nuevo_resultado_formato_fichero = []

        fichero_txt = open(f'{path_fichero_txt}', 'w')

        for box in nuevo_resultado:
            xmax = int(box[2])
            ymax = int(box[3])
            xmin = int(box[0])
            ymin = int(box[1])

            lista_nuevo_resultado_formato_fichero.append(
                f"{xmin} {ymin} {xmax} {ymax}")

        for linea in lineas:
            if linea.strip("\n") not in lista_nuevo_resultado_formato_fichero:
                fichero_txt.write(linea)

        fichero_txt.close()

        # Sobreescribir imagen. Tenemos que coger la imagen original, no la que esta pintada.
        imagen = cv2.imread(path_archivo[:-7]+'.png', cv2.IMREAD_COLOR)

        boxes_existentes = []

        with open(f'{path_fichero_txt}') as texto:
            for indice, linea in enumerate(texto):
                if indice == 0:
                    continue

                linea = linea.split(" ", 3)
                boxes_existentes.append(
                    [int(linea[0]), int(linea[1]), int(linea[2]), int(linea[3])])

            for box in boxes_existentes:
                xmax = int(box[2])
                ymax = int(box[3])
                xmin = int(box[0])
                ymin = int(box[1])

                cv2.rectangle(imagen, (xmin, ymin),
                              (xmax, ymax), (255, 0, 0), 2)

        cv2.imwrite(f"{path_archivo}", imagen)
        fichero_txt.close()

    elif flag_edicion_clasificacion:
        # Sobreescribir fichero txt
        fichero_txt = open(f'{path_fichero_txt}', 'w')
        fichero_txt.write(f'{nuevo_resultado}')
        fichero_txt.close()
