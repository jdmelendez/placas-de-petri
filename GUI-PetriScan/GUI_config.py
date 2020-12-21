# =============================================== PATHS =============================================================

PATH_MODELOS = "C:/Users/jmelendez/GIT/placas-de-petri/models_trained"
# PATH_IMGS_PRUEBA = "C:/Users/JoseD/OneDrive/Desktop/REPOSITORIO DE CARPETAS/PROYECTOS/PetriNet/interface/Prueba"
PATH_IMGS_PRUEBA = "C:/Users/jmelendez/GIT/placas-de-petri/GUI-PetriScan/imgs_prueba"

# ============================================ MODELOS ===============================================================

# Se indica el identificador general, la ruta del modelo, el tipo de tecnica, y el tipo de transformacion de imagen.

MODELOS = {
    # Modelos de challenge
    "CH_AB": [f"{PATH_MODELOS}/330-CH-AB.pt", "DETECCION", "A"],
    "CH_BC": [f"{PATH_MODELOS}/429-CH-BC.pt", "DETECCION", "A"],
    "CH_CA": [f"{PATH_MODELOS}/47-CH-CA.pt", "DETECCION", "A"],
    "CH_EC": [f"{PATH_MODELOS}/376-CH-EC.pt", "DETECCION", "A"],
    "CH_PA": [f"{PATH_MODELOS}/668-CH-PA.pt", "DETECCION", "A"],
    "CH_SA": [f"{PATH_MODELOS}/870-CH-SA.pt", "DETECCION", "A"],

    # Modelos de aguas
    "AG_BC": [f"{PATH_MODELOS}/797-AG-BC.pt", "CLASIFICACION", "B"]

}

# ================================================== DATAFRAME =======================================================
NOMBRE_COLUMNAS = ["ID_IMAGEN",
                   "ID_ANALISIS",
                   "ID_PATOGENO",
                   "ID_MODELO",
                   "PATH_IMAGEN"]

# =================================================== CLASES ==========================================================

CLASES_CLASIFICACION = ["AUSENCIA", "PRESENCIA"]

# ============================================ NOMBRES ID IMAGEN NUEVA ==============================================

# Se define un diccionario para que al clickar el boton del patogeno, se asocie su nombre con un identificador para
# poner nombre a los ficheros. Si se cambia el texto de los botones, se debe de cambiar aqui tambien.

TEXTO_BOTONES_PATOGENO = {
    "ASPERGILLUS\nBRASILIENSIS": "AB",
    "BURKHOLDERIA\nCEPACIA": "BC",
    "CANDIDA\nALBICANS": "CA",
    "ESCHERICHIA\nCOLI": "EC",
    "PSEUDOMONAS\nAERUGINOSA": "PA",
    "STAPHYLOCOCCUS\nAUREUS": "SA",
    "MOHOS": "MO",
    "LEVADURAS": "LE",
    "AEROBIOS\nMESOFILOS": "AM"
}

# Si se cambia el nombre de las carpetas se ha de cambiar tambien aqui.
TEXTO_CARPETAS_ANALISIS = {
    "Aguas": "AG",
    "Challenge Test": "CH",
    "Superficies": "SUP",
    "Ambientes": "AMB",
    "Producto Terminado": "PT",
    "Producto Quimico": "PQ"
}

TEXTO_BOTONES_DILUCION = {
    "-1": "D1",
    "-2": "D2",
    "-3": "D3"
}

TEXTO_COMBOBOX_DILUCION = {
    "-4": "D4",
    "-5": "D5",
    "-6": "D6",
    "-7": "D7"
}

TEXTO_BOTONES_TIEMPO = {
    "T0": "T0",
    "T7": "T7",
    "T14": "T14",
    "T28": "T28"
}

TEXTO_COMBOBOX_ZONAS = {
    "PASILLO": "PAS",
    "PARED": "PAR",
    "SUELO": "SUE"
}


# ============================================  SQL ==============================================

SERVER_NAME = 'SRVBDI'
DATABASE_NAME = 'PLACASPETRI'
USERNAME = 'userplacaspetri'
PASSWORD = 'userplacaspetri2020'
CONN_STRING = f'DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={SERVER_NAME};DATABASE={DATABASE_NAME};UID={USERNAME}; PWD={PASSWORD}'
