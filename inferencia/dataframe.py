from config import NOMBRE_COLUMNAS
import pandas as pd


def crear_dataframe_vacio():
    df = pd.DataFrame(columns=NOMBRE_COLUMNAS)

    return df


def rellenar_dataframe(df, nombre_columna, objeto):

    # df_aux = pd.DataFrame(objeto, columns=[nombre_columna])
    # df = df.append(df_aux)

    df[nombre_columna] = objeto

    return df
