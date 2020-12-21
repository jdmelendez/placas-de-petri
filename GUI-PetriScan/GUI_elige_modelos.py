from pathlib import Path
from GUI_dataframe import rellenar_dataframe
from GUI_config import MODELOS, NOMBRE_COLUMNAS


def elige_modelos(df, paths_imagenes):

    id_imagen_list = [(Path(path_imagen).stem)[0:5]
                      for path_imagen in paths_imagenes]
    id_analisis_list = [id_imagen.split('_')[0]
                        for id_imagen in id_imagen_list]
    id_patogeno_list = [id_imagen.split('_')[1]
                        for id_imagen in id_imagen_list]
    id_modelos_list = [(Path(MODELOS[id_imagen][0]).stem)
                       for id_imagen in id_imagen_list]

    modelos_a_utilizar = (dict.fromkeys(id_imagen_list)).keys() | set()

    df_new = rellenar_dataframe(
        df=df, nombre_columna=NOMBRE_COLUMNAS[0], objeto=id_imagen_list)
    df_new = rellenar_dataframe(
        df=df_new, nombre_columna=NOMBRE_COLUMNAS[1], objeto=id_analisis_list)
    df_new = rellenar_dataframe(
        df=df_new, nombre_columna=NOMBRE_COLUMNAS[2], objeto=id_patogeno_list)
    df_new = rellenar_dataframe(
        df=df_new, nombre_columna=NOMBRE_COLUMNAS[3], objeto=id_modelos_list)
    df_new = rellenar_dataframe(
        df=df_new, nombre_columna=NOMBRE_COLUMNAS[-1], objeto=paths_imagenes)

    return modelos_a_utilizar, id_imagen_list, df_new
