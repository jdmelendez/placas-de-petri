from config import MODELOS


def flags_clasificacion_deteccion(modelos_a_utilizar):

    FLAG_LISTA = [MODELOS[id][1] for id in modelos_a_utilizar]

    FLAG_CLASIFICACION = 1 if "CLASIFICACION" in FLAG_LISTA else 0
    FLAG_DETECCION = 1 if "DETECCION" in FLAG_LISTA else 0

    return FLAG_CLASIFICACION, FLAG_DETECCION
