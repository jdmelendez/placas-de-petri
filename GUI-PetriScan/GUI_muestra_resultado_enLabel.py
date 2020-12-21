from GUI_config import CLASES_CLASIFICACION


def muestra_resultado_enLabel(path_imagen):
    path_fichero = path_imagen[:-3] + "txt"

    fichero = open(path_fichero, 'r')
    primera_linea = fichero.readline()

    if primera_linea in CLASES_CLASIFICACION:
        resultado = primera_linea
    else:
        cantidad_lineas_fichero = len(fichero.readlines())
        resultado = f"{cantidad_lineas_fichero} colonias"

    return resultado
