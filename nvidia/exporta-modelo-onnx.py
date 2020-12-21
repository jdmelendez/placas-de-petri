import io
import numpy as np
import torch.onnx
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

path_modelo_deteccion = "./models_trained/544-CH-CA.pt"
path_modelo_clasificacion = "./models_trained/797-AG-BC.pt"

FLAG_DETECCION = 1
FLAG_CLASIFICACION = 0

device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')

# ====================== MODELO DE DETECCION ==================================


def define_modelo_deteccion():

    # Definiendo el modelo original y cambiando sus parametros
    model_FasterRCNN = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        pretrained=True)
    model_FasterRCNN_in_features = model_FasterRCNN.roi_heads.box_predictor.cls_score.in_features
    model_FasterRCNN.roi_heads.box_predictor = FastRCNNPredictor(
        model_FasterRCNN_in_features, 3)
    model_FasterRCNN.roi_heads.detections_per_img = 500
    factor = 1
    model_FasterRCNN.rpn.anchor_generator.sizes = (
        (16*factor,), (32*factor,), (64*factor,), (128*factor,), (256*factor,))
    model_FasterRCNN.transform.min_size = (1232,)
    model_FasterRCNN.transform.max_size = 1232

    return model_FasterRCNN


# ====================== MODELOS DE CLASIFICACIÓN ==================================

def define_modelo_clasificacion():

    model_resnet18 = torchvision.models.resnet18(pretrained=True)
    # 2 = Numero de clases (AUSENCIA, PRESENCIA)
    model_resnet18_num_ftrs = model_resnet18.fc.in_features
    model_resnet18.fc = torch.nn.Linear(model_resnet18_num_ftrs, 2)

    return model_resnet18

##############################################################################################


if FLAG_CLASIFICACION:

    # Carga del modelo original
    modelo = define_modelo_clasificacion()
    modelo.load_state_dict(torch.load(
        path_modelo_clasificacion, map_location=device))
    modelo.eval()

    entrada_modelo_torch = torch.randn(
        1, 3, 128, 128, requires_grad=True)

    # salida_modelo_torch = modelo(entrada_modelo_torch)

    # Exportamos el modelo a ONNX
    path_modelo_onnx = f"{path_modelo_clasificacion[:-2]}" + 'onnx'
    torch.onnx.export(modelo, entrada_modelo_torch,
                      path_modelo_onnx,
                      export_params=True,
                      opset_version=10,
                      do_constant_folding=True,
                      input_names=['input'],
                      output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})


if FLAG_DETECCION:

    # Carga del modelo original
    modelo = define_modelo_deteccion()
    modelo.load_state_dict(torch.load(
        path_modelo_deteccion, map_location=device))
    modelo.eval()

    x = torch.rand(3, 1232, 1232,dtype=torch.float32)
    entrada_modelo_torch = x.clone().detach().requires_grad_(True)

    # salida_modelo_torch = modelo([entrada_modelo_torch])

    # Exportamos el modelo a ONNX
    path_modelo_onnx = (f"{path_modelo_deteccion[:-2]}" + 'onnx')#																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																								.format(32,1)
    torch.onnx.export(modelo, [entrada_modelo_torch], path_modelo_onnx,opset_version=11)#,export_params=True, keep_initializers_as_inputs=True)
