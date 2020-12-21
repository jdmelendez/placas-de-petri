import io
import numpy as np
import torch.onnx
import onnx
import onnxruntime
from torchvision import transforms
from PIL import Image
from torchvision.ops._register_onnx_ops import _onnx_opset_version
from io import BytesIO


def pipeline_ONXX(path_modelo, modelo, dim_img, deteccion=0, imagen=None):
    # entrada_modelo_torch, salida_modelo_torch, path_modelo_onnx = exportar_modelo_ONNX(
    #     path_modelo, modelo, dim_img, deteccion, imagen)
    # validar_modelo_ONNX(path_modelo_onnx)
    # sesion_runtime = runtime_modelo_ONNX(path_modelo_onnx)
    # evaluar_modelo_ONNX(
    #     sesion_runtime, entrada_modelo_torch, salida_modelo_torch)
    # prediccion = inferencia_ONNX(sesion_runtime, imagen)
    path_modelo_onnx = f"{path_modelo[:-2]}" + 'onnx'
    return path_modelo_onnx


def exportar_modelo_ONNX(path_modelo, modelo, dim_img, deteccion, imagen):

    batch_size = 1

    # Cargamos los pesos del modelo
    def map_location(storage, loc): return storage
    if torch.cuda.is_available():
        map_location = None

    device = torch.device('cpu')
    modelo.load_state_dict(torch.load(path_modelo, map_location=device))

    # Ponemos el modelo en modo inferencia
    modelo.eval()

    # Creamos el vector de entrada al modelo
    if deteccion:

        path_modelo_onnx = f"{path_modelo[:-2]}" + 'onnx'

        x = torch.rand(3, 1232, 1232)
        img = x.clone().detach().requires_grad_(True)

        salida_modelo_torch = modelo([img])
        torch.onnx.export(modelo, [img], path_modelo_onnx, opset_version=11,
                          export_params=True, keep_initializers_as_inputs=True)
        entrada_modelo_torch = [img]

    else:
        entrada_modelo_torch = torch.randn(
            batch_size, 3, dim_img, dim_img, requires_grad=True)

        salida_modelo_torch = modelo(entrada_modelo_torch)

        # Exportamos el modelo a ONNX
        path_modelo_onnx = f"{path_modelo[:-2]}" + 'onnx'
        torch.onnx.export(modelo, entrada_modelo_torch,
                          path_modelo_onnx,
                          export_params=True,
                          opset_version=10+deteccion,
                          do_constant_folding=True,
                          input_names=['input'],
                          output_names=['output'],
                          dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})

    return entrada_modelo_torch, salida_modelo_torch, path_modelo_onnx


def validar_modelo_ONNX(path_modelo_onnx):
    modelo_onnx = onnx.load(path_modelo_onnx)
    onnx.checker.check_model(modelo_onnx)


def runtime_modelo_ONNX(path_modelo_onnx):
    sesion_runtime = onnxruntime.InferenceSession(path_modelo_onnx)

    return sesion_runtime


def evaluar_modelo_ONNX(sesion_runtime, entrada_modelo_torch, salida_modelo_torch):

    entradas_sesion_runtime = {
        sesion_runtime.get_inputs()[0].name: to_numpy(entrada_modelo_torch)}
    salidas_sesion_runtime = sesion_runtime.run(None, entradas_sesion_runtime)

    np.testing.assert_allclose(to_numpy(
        salida_modelo_torch), salidas_sesion_runtime[0], rtol=1e-03, atol=1e-05)


def inferencia_ONNX(imagen, sesion_runtime,  deteccion):
    # sesion_runtime = onnxruntime.InferenceSession(path_modelo_onnx)

    if deteccion == 0:
        entradas_sesion_runtime = {
            sesion_runtime.get_inputs()[0].name: to_numpy(imagen)}
        salidas_sesion_runtime = sesion_runtime.run(
            None, entradas_sesion_runtime)
        print(salidas_sesion_runtime)
        prediccion = salidas_sesion_runtime[0]
    else:
        imagen = imagen.numpy()
        # entradas_sesion_runtime = {
        #     sesion_runtime.get_inputs()[0].name: to_numpy(imagen)}
        salidas_sesion_runtime = sesion_runtime.run(
            None, {"image.1": imagen})
        prediccion = salidas_sesion_runtime
        # print(prediccion)

    return prediccion


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
