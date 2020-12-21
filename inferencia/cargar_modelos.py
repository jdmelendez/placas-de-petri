import torch
from config import MODELOS
from definir_modelos import define_modelo_deteccion, define_modelo_clasificacion
from flags_clasificacion_deteccion import flags_clasificacion_deteccion
from modelo_ONNX import pipeline_ONXX
import onnx
import onnxruntime


def cargar_modelos(modelos_a_utilizar, FLAG_ONNX, imagenes_lista=None):
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

    modelos = {k: [(pipeline_carga_modelos_deteccion(k, arquitectura_modelo_deteccion, device, FLAG_ONNX, imagenes_lista) if v[1] == "DETECCION" else pipeline_carga_modelos_clasificacion(k, arquitectura_modelo_clasificacion, device, FLAG_ONNX)), v[1]] for (
        k, v) in MODELOS.items() if k in modelos_a_utilizar}

    return modelos, device


def pipeline_carga_modelos_deteccion(id_modelo, arquitectura_modelo, device, ONNX, imagen=None):

    if ONNX == 0:
        modelo = arquitectura_modelo
        path_modelo = (MODELOS[id_modelo][0])

        modelo.load_state_dict(
            torch.load(path_modelo, map_location=device))
        modelo.to(device)
        modelo.eval()

    else:
        modelo = arquitectura_modelo
        path_modelo = (MODELOS[id_modelo][0])
        path_modelo_onnx = pipeline_ONXX(
            path_modelo, modelo, 1232, 1, imagen[0])
        modelo = onnxruntime.InferenceSession(path_modelo_onnx)

    return modelo


def pipeline_carga_modelos_clasificacion(id_modelo, arquitectura_modelo, device, ONNX):

    if ONNX == 0:
        modelo = arquitectura_modelo
        path_modelo = (MODELOS[id_modelo][0])

        modelo.load_state_dict(
            torch.load(path_modelo, map_location=device))
        modelo.to(device)
        modelo.eval()

    if ONNX:
        modelo = arquitectura_modelo
        path_modelo = (MODELOS[id_modelo][0])
        path_modelo_onnx = pipeline_ONXX(path_modelo, modelo, 128)
        modelo = onnxruntime.InferenceSession(path_modelo_onnx)

    return modelo
