
from dataframe import crear_dataframe_vacio
from cargar_modelos import cargar_modelos
from predecir import predecir
from crear_dataset import crear_dataset
from resultados_plot import resultados_plot
from config import PATH_IMGS_PRUEBA


# ================================================= DATAFRAME ======================================================

# Se crea un dataframe vacio para ir almacenando los datos y comprobar que todo funciona correctamente.
df = crear_dataframe_vacio()


# ================================================= DATASET ======================================================

# Se obtiene una lista matricial con las imagenes, y una lista con las rutas de las imagenes
imagenes_lista, paths_imagenes, modelos_a_utilizar, id_imagenes_lista, df = crear_dataset(
    df, path_imagenes=PATH_IMGS_PRUEBA)


# ===============================================CARGA DE MODELOS =================================================

# Se instancian los modelos de red que se van a utilizar en la prediccion.
# modelos, device = cargar_modelos(modelos_a_utilizar, 0)


# # # ================================================= PREDICCION NORMAL==============================================

# # Se predice el resultado en funcion de la imagen que le pases, bien puede ser de clasificacion, bien de deteccion.
# print("--- PREDICION NORMAL ---")
# predicciones = [predecir(
#     modelos=modelos, imagen=imagen, device=device, indice=indice, id_imagenes_lista=id_imagenes_lista, FLAG_ONNX=0)
#     for indice, imagen in enumerate(imagenes_lista)]
# print(predicciones)


# ================================================= PREDICCION ONNX =============================================
# Se instancian los modelos de red que se van a utilizar en la prediccion.
print("\n--- PREDICION ONNX ---")
modelos, device = cargar_modelos(
    modelos_a_utilizar, 1, imagenes_lista=imagenes_lista)
predicciones = [predecir(
    modelos=modelos, imagen=imagen, device=device, indice=indice, id_imagenes_lista=id_imagenes_lista, FLAG_ONNX=1)
    for indice, imagen in enumerate(imagenes_lista)]

# ================================================= RESULTADOS ======================================================

# Se muestran imagenes con los resultados, tanto de clasificacion, como de deteccion.

resultados_plot(paths_imagenes=paths_imagenes, predicciones=predicciones)
