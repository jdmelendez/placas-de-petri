from GUI_dataframe import crear_dataframe_vacio
from GUI_cargar_modelos import cargar_modelos
from GUI_predecir import predecir
from GUI_crear_dataset import crear_dataset

from GUI_crea_mensaje_alerta import crea_mensaje_alerta


def pipeline_inferencia(path_imagenes):  # , barra):
    # Se crea un dataframe vacio para ir almacenando los datos y comprobar que todo funciona correctamente.
    df = crear_dataframe_vacio()


# ================================================= DATASET ======================================================

# Se obtiene una lista matricial con las imagenes, y una lista con las rutas de las imagenes
    try:
        imagenes_lista, paths_imagenes, modelos_a_utilizar, id_imagenes_lista, df = crear_dataset(
            df, path_imagenes=path_imagenes)

    except:
        return


# ===============================================CARGA DE MODELOS =================================================

# Se instancian los modelos de red que se van a utilizar en la prediccion.
    modelos, device = cargar_modelos(modelos_a_utilizar)

    return modelos, device, id_imagenes_lista, paths_imagenes, imagenes_lista

# ================================================= PREDICCION ======================================================

# Se predice el resultado en funcion de la imagen que le pases, bien puede ser de clasificacion, bien de deteccion.
#  barra.setValue((indice+1)*100/len(imagenes_lista))
    # predicciones = {paths_imagenes[indice]: predecir(
    #     modelos=modelos, imagen=imagen, device=device, indice=indice, id_imagenes_lista=id_imagenes_lista)
    #     for indice, imagen in enumerate(imagenes_lista)}

    # return predicciones
# ================================================= RESULTADOS ======================================================

# Se muestran imagenes con los resultados, tanto de clasificacion, como de deteccion.

    # resultados_plot(paths_imagenes=paths_imagenes,
    #                 predicciones=predicciones)
