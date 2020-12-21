import torch
from GUI_config import MODELOS
from GUI_definir_modelos import define_modelo_deteccion, define_modelo_clasificacion
from GUI_flags_clasificacion_deteccion import flags_clasificacion_deteccion


def cargar_modelos(modelos_a_utilizar):
    """[Se comprueban si van a haber modelos de deteccion, de clasificacion o ambos, y en funciones de esto]

    Args:
        FLAG_CLASIFICACION ([type]): [description]
        FLAG_DETECCION ([type]): [description]
        modelos_a_utilizar ([type]): [description]

    Returns:
        [type]: [description]
    """

    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    FLAG_CLASIFICACION, FLAG_DETECCION = flags_clasificacion_deteccion(
        modelos_a_utilizar)

    if FLAG_DETECCION:
        # Adquirimos la arquitectura del modelo de deteccion
        arquitectura_modelo_deteccion = define_modelo_deteccion()

    if FLAG_CLASIFICACION:
        # Adquirimos la arquitectura del modelo de clasificacion
        arquitectura_modelo_clasificacion = define_modelo_clasificacion()

    # Asociamos la arquitectura, y cargamos los pesos de los modelos a utilizar
    # modelos = {}

    # modelos = {k: [pipeline_carga_modelos_deteccion(k, arquitectura_modelo_deteccion, device), v[1]] for (
    #     k, v) in MODELOS.items() if k in modelos_a_utilizar}

    modelos = {k: [(pipeline_carga_modelos_deteccion(k, arquitectura_modelo_deteccion, device) if v[1] == "DETECCION" else pipeline_carga_modelos_clasificacion(k, arquitectura_modelo_clasificacion, device)), v[1]] for (
        k, v) in MODELOS.items() if k in modelos_a_utilizar}

    return modelos, device


def pipeline_carga_modelos_deteccion(id_modelo, arquitectura_modelo, device):

    modelo = arquitectura_modelo
    modelo.load_state_dict(
        torch.load(MODELOS[id_modelo][0], map_location=device))
    modelo.to(device)
    modelo.eval()

    return modelo


def pipeline_carga_modelos_clasificacion(id_modelo, arquitectura_modelo, device):

    modelo = arquitectura_modelo
    modelo.load_state_dict(
        torch.load(MODELOS[id_modelo][0], map_location=device))
    modelo.to(device)
    modelo.eval()

    return modelo
