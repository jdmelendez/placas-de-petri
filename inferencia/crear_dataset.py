import glob
from config import MODELOS
from elige_modelos import elige_modelos
import transformar_dataset


# transformacion=aplica_transformacion):
def crear_dataset(df, path_imagenes,  transformaciones=transformar_dataset):
    """[Genera el dataset sobre el que se haran las preddiciones partiendo de la ruta en la cual se encuentran las imagenes.
    Ademas, se aplican las transformaciones necesarias (tensor, dimension, etc.) a cada imagen, y se obtiene un diccionariocon los modelos que se van a utilizar]

    Args:
        df ([type]): [dataframe en el cual se alojará la informacion de las imagenes]
        path_imagenes ([type], optional): [Ruta donde se encuentras las imagenes establecida en el archivo "config.py"]. Defaults to PATH_IMGS_PRUEBA.
        transformaciones ([type], optional): [diccionario de funciones con cada tipo de transformacion aplicada]. Defaults to transformar_dataset.

    Returns:
        [type]: [se devuelve una lista con las matrices de las imagenes transformadas, una lista con las rutas de cada imagen,
         un diccionario con los modelos a utilizar, una lista con los identificadores de cada imagen, el dataframe con la informacion]
    """

    # EN CASO DE QUERER HACER INFERENCIA DE TODA LA CARPETA # TODO
    if path_imagenes[-3:] == 'png':
        lista_paths_imgs = [path_imagenes]
    else:
        lista_paths_imgs = glob.glob(f"{path_imagenes}/*.png")

    modelos_a_utilizar, id_imagenes_lista, df = elige_modelos(
        df, lista_paths_imgs)

    lista_imagenes = [transformaciones.__dict__.get(MODELOS[id_imagenes_lista[indice]][2])(imagen)
                      for indice, imagen in enumerate(lista_paths_imgs)]

    return lista_imagenes, lista_paths_imgs, modelos_a_utilizar, id_imagenes_lista, df
