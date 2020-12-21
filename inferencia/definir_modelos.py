import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


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
