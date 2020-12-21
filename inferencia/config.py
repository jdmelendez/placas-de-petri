# =============================================== PATHS =============================================================

PATH_MODELOS = "C:/Users/jmelendez/GIT/placas-de-petri/models_trained"
PATH_IMGS_PRUEBA = "C:/Users/jmelendez/GIT/placas-de-petri/inferencia/imgs_prueba"


# ============================================ MODELOS ===============================================================

# Se indica el identificador general, la ruta del modelo, el tipo de tecnica, y el tipo de transformacion de imagen.

MODELOS = {
    # Modelos de challenge
    "CH_AB": [f"{PATH_MODELOS}/323-CH-AB.pt", "DETECCION", "A"],
    "CH_BC": [f"{PATH_MODELOS}/993-CH-BC.pt", "DETECCION", "A"],
    "CH_CA": [f"{PATH_MODELOS}/544-CH-CA.pt", "DETECCION", "A"],
    "CH_EC": [f"{PATH_MODELOS}/610-CH-EC.pt", "DETECCION", "A"],
    "CH_PA": [f"{PATH_MODELOS}/938-CH-PA.pt", "DETECCION", "A"],
    "CH_SA": [f"{PATH_MODELOS}/600-CH-SA.pt", "DETECCION", "A"],

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


# =================================================== ONXX ==========================================================
FLAG_ONNX = True
