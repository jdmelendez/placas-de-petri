print("\nIniciando aplicacion...")
'''
________________________________________________________________________________________________________________________
                                                    PARÁMETROS
________________________________________________________________________________________________________________________
'''

# Parametros filtro
iou_threshold = 0.3
dist_eucl_threshold = 20

'''
________________________________________________________________________________________________________________________
                                                  LIBRERIAS
________________________________________________________________________________________________________________________
'''

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch.nn.functional as F
from torchvision.transforms import functional as T
from torch import nn, random
from torchvision import datasets, transforms, models

import numpy as np
import math
import time
import glob
import matplotlib.pyplot as plt
from matplotlib import patches
import os
from PIL import Image

'''
________________________________________________________________________________________________________________________
                                                    FILTRO
________________________________________________________________________________________________________________________
'''


def filter_boxes(boxes, iou_threshold, dist_eucl_threshold):
    boxes_filt = boxes
    deletes = 0
    indices_delete = []
    indice_i = 0

    for i in boxes:
        x1A = i[0]
        y1A = i[1]
        x2A = i[2]
        y2A = i[3]

        indice_j = 0
        for j in boxes[indice_i:]:
            iou = 0
            dist_eucl = 1000
            x1B = j[0]
            y1B = j[1]
            x2B = j[2]
            y2B = j[3]

            # Arriba izquierda - Abajo derecha
            if ((x1A < x1B) and (y1A < y1B) and (x2A < x2B) and (y2A < y2B) and (x1B < x2A) and (y1B < y2A)) or (
                    (x1A > x1B) and (y1A > y1B) and (x2A > x2B) and (y2A > y2B) and (x1A < x2B) and (y1A < y2B)):
                x_left = max(x1A, x1B)
                y_top = max(y1A, y1B)
                x_right = min(x2A, x2B)
                y_bottom = min(y2A, y2B)

            # Arriba derecha
            elif ((x1A < x1B) and (y1A > y1B) and (x2A < x2B) and (y2A > y2B) and (x1B < x2A) and (y1A < y2B)):
                x_left = x1B
                y_top = y1A
                x_right = x2A
                y_bottom = y2B

            # Abajo izquierda
            elif ((x1A > x1B) and (y1A < y1B) and (x2A > x2B) and (y2A < y2B) and (x1A < x2B) and (y1B < y2A)):
                x_left = x1A
                y_top = y1B
                x_right = x2B
                y_bottom = y2A

            # Arriba
            elif ((x1A < x1B) and (y1A > y1B) and (x2A > x2B) and (y2A > y2B) and (y1A < y2B) and (x1B < x2A)):
                x_left = x1B
                y_top = y1A
                x_right = x2B
                y_bottom = y2B

            # Derecha
            elif ((x1A < x1B) and (y1A < y1B) and (x2A < x2B) and (y2A > y2B) and (y1A < y2B) and (x1B < x2A)):
                x_left = x1B
                y_top = y1B
                x_right = x2A
                y_bottom = y2B

            # Abajo
            elif ((x1A < x1B) and (y1A < y1B) and (x2A > x2B) and (y2A < y2B) and (y1A < y2B) and (x1B < x2A)):
                x_left = x1B
                y_top = y1B
                x_right = x2B
                y_bottom = y2A

            # Izquierda
            elif ((x1A > x1B) and (y1A < y1B) and (x2A > x2B) and (y2A > y2B) and (y1A < y2B) and (x1B < x2A)):
                x_left = x1A
                y_top = y1B
                x_right = x2B
                y_bottom = y2B

            # Arriba ancho
            elif ((x1A > x1B) and (y1A > y1B) and (x2A < x2B) and (y2A > y2B) and (y1A > y2B) and (x1A < x2B)):
                x_left = x1A
                y_top = y1A
                x_right = x2A
                y_bottom = y2B

            # Bajo ancho
            elif ((x1A > x1B) and (y1A < y1B) and (x2A < x2B) and (y2A < y2B) and (y1A < y2B) and (x1A < x2B)):
                x_left = x1A
                y_top = y1B
                x_right = x2A
                y_bottom = y2A

            # Izquierda ancho
            elif ((x1A > x1B) and (y1A > y1B) and (x2A > x2B) and (y2A < y2B) and (y1A < y2B) and (x1A < x2B)):
                x_left = x1A
                y_top = y1A
                x_right = x2B
                y_bottom = y2A

            # Derecha ancho
            elif ((x1A < x1B) and (y1A > y1B) and (x2A < x2B) and (y2A < y2B) and (y1A < y2B) and (x1A < x2B)):
                x_left = x1B
                y_top = y1A
                x_right = x2A
                y_bottom = y2A

            # Horizontal
            elif ((x1A > x1B) and (y1A < y1B) and (x2A < x2B) and (y2A > y2B) and (y1A < y2B) and (x1A < x2B)):
                x_left = x1A
                y_top = y1B
                x_right = x2A
                y_bottom = y2B

            # Vertical
            elif ((x1A < x1B) and (y1A > y1B) and (x2A > x2B) and (y2A < y2B) and (y1A < y2B) and (x1A < x2B)):
                x_left = x1B
                y_top = y1A
                x_right = x2B
                y_bottom = y2A


            # Dentro
            elif (x1A < x1B and y1A < y1B and x2A > x2B and y2A > y2B) or (
                    (x1A > x1B and y1A > y1B and x2A < x2B and y2A < y2B)):
                centroAx = x1A + (x2A - x1A) / 2
                centroAy = y1A + (y2A - y1A) / 2
                centroBx = x1B + (x2B - x1B) / 2
                centroBy = y1B + (y2B - y1B) / 2
                dist_eucl = math.sqrt((centroAx - centroBx) ** 2 + (centroAy - centroBy) ** 2)

                if dist_eucl < dist_eucl_threshold:
                    boxes_filt[indice_i + indice_j][:] = [0, 0, 0, 0]
                indice_j += 1
                continue

            else:
                indice_j += 1
                continue

            # The intersection of two axis-aligned bounding boxes
            intersection_area = (x_right - x_left) * (y_bottom - y_top)

            # compute the area of both AABBs
            bb1_area = (x2A - x1A) * (y2A - y1A)
            bb2_area = (x2B - x1B) * (y2B - y1B)

            # compute the intersection over union
            iou = intersection_area / float(bb1_area + bb2_area - intersection_area)

            if (iou > iou_threshold):
                boxes_filt[indice_i + indice_j][:] = [0, 0, 0, 0]

            indice_j += 1
        indice_i += 1

    for i in range(len(boxes_filt)):
        result = np.all((boxes_filt[i] == 0))
        if result:
            indices_delete.append(i)

    boxes_filt = np.delete(boxes_filt, indices_delete, axis=0)

    return boxes_filt


'''
________________________________________________________________________________________________________________________
                                           DEFINICION MODELO DETECCIÓN
________________________________________________________________________________________________________________________
'''
print("Definiendo modelos...")

model_CH_CA = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model_CH_PA = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model_CH_BC = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model_CH_EC = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model_CH_SA = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model_CH_AB = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model_AG_AM = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

# Cantidad de clases que tenemos (background + colonia)
num_classes = 2

# Obtenemos el numero de caracteristicas de entrada al clasificador
in_features = model_CH_CA.roi_heads.box_predictor.cls_score.in_features

# Reemplazamos la cabecera de la red pre-trained por la nueva
model_CH_CA.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
model_CH_PA.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
model_CH_BC.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
model_CH_EC.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
model_CH_SA.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
model_CH_AB.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
model_AG_AM.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Cantidad maxima de objetos a detectar
model_CH_CA.roi_heads.detections_per_img = 400
model_CH_PA.roi_heads.detections_per_img = 400
model_CH_BC.roi_heads.detections_per_img = 400
model_CH_SA.roi_heads.detections_per_img = 400
model_CH_EC.roi_heads.detections_per_img = 400
model_CH_AB.roi_heads.detections_per_img = 400
model_AG_AM.roi_heads.detections_per_img = 400

'''
________________________________________________________________________________________________________________________
                                       DEFINICION MODELOS CLASIFICACION
________________________________________________________________________________________________________________________
'''


class Net(nn.Module):
    def __init__(self):
        global conv_fin, dim_img_fin
        super(Net, self).__init__()

        # Capa convolucional de entrada (profundidad imagen entrada, profundidad imagen salida, kernel, padding, stride) --> 800 x 800 x 3
        conv_ini = 128
        self.conv1 = nn.Conv2d(3, conv_ini, 3, padding=1)  # TODO
        conv_fin = conv_ini
        dim_img_fin = int(conv_ini) / 2

        # Capa convolucional --> 400 x 400 x 64
        self.conv2 = nn.Conv2d(conv_ini, 2 * conv_ini, 3, padding=1)  # TODO
        conv_fin = 2 * conv_ini
        dim_img_fin = int(conv_ini / 4)

        # Capa convolucional --> 200 x 200 x 128
        self.conv3 = nn.Conv2d(2 * conv_ini, 4 * conv_ini, 3, padding=1)  # TODO
        conv_fin = 4 * conv_ini
        dim_img_fin = int(conv_ini / 8)

        # Max pooling (Kernel, stride)
        self.pool = nn.MaxPool2d(2, 2)  # TODO

        # Capa fully conected
        self.fc1 = nn.Linear(conv_fin * dim_img_fin * dim_img_fin, 1000)  # TODO

        # Capa fully conected salida (tamaño, numero de clases)
        self.fc2 = nn.Linear(1000, 2)  # TODO

        # Abandono --> Evitar overfitting
        self.dropout = nn.Dropout(0.25)  # TODO

    def forward(self, x):
        # Secuencia de capas convolucionales y max pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        # Se aplana la matriz --> Vector
        x = x.view(-1, conv_fin * dim_img_fin * dim_img_fin)  # TODO

        # Capa de abandono
        x = self.dropout(x)

        # Capa oculta con Relu
        x = F.relu(self.fc1(x))

        # Capa de abandono
        x = self.dropout(x)

        # Capa oculta de salida con Relu
        x = self.fc2(x)

        return x


model_AG_PA = models.resnet18(pretrained=True)
num_ftrs = model_AG_PA.fc.in_features
model_AG_PA.fc = nn.Linear(num_ftrs, 2)

model_AG_BC = Net()
# model_AG_PA = ResNet18()

'''
________________________________________________________________________________________________________________________
                                             CARGA MODELOS
________________________________________________________________________________________________________________________
'''
print("Cargando modelos...")

# COMPROBAMOS SI EXISTE GPU Y MOVEMOS EL MODELO AL DISPOSITIVO:
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

id_CH_CA = "779-CH-CA"
id_CH_PA = "571-CH-PA"
id_CH_BC = "139-CH-BC"
id_CH_EC = "519-CH-EC"
id_CH_SA = "963-CH-SA"
id_CH_AB = "399-CH-AB"
id_AG_BC = "340-AG-BC"
id_AG_PA = "964-AG-PA"
# id_AG_AM = "964-AG-PA"

path_models = "C:/Users/jmelendez/OneDrive - RNBCOSMETICOS/Escritorio/Proyecto_Placas de Petri/Codigo/Inferencia/Modelos Entrenados"
path_model_CH_CA = os.path.join(path_models, "Challenge Test/Candida Albicans")
path_model_CH_BC = os.path.join(path_models, "Challenge Test/Burkholderia Cepacia")
path_model_CH_PA = os.path.join(path_models, "Challenge Test/Pseudomonas Aeruginosa")
path_model_CH_EC = os.path.join(path_models, "Challenge Test/Escherichia Coli")
path_model_CH_AB = os.path.join(path_models, "Challenge Test/Aspergillus Brasiliensis")
path_model_CH_SA = os.path.join(path_models, "Challenge Test/Staphylococcus Aureus")

path_model_AG_BC = os.path.join(path_models, "Aguas/Burkholderia Cepacia")
path_model_AG_PA = os.path.join(path_models, "Aguas/Pseudomonas Aeruginosa")
path_model_AG_AM = os.path.join(path_models, "Aguas/Aerobios Mesofilos")

model_CH_CA.load_state_dict(torch.load(f"{path_model_CH_CA}/{id_CH_CA}.pt", map_location=device))
model_CH_PA.load_state_dict(torch.load(f"{path_model_CH_PA}/{id_CH_PA}.pt", map_location=device))
model_CH_BC.load_state_dict(torch.load(f"{path_model_CH_BC}/{id_CH_BC}.pt", map_location=device))
model_CH_EC.load_state_dict(torch.load(f"{path_model_CH_EC}/{id_CH_EC}.pt", map_location=device))
model_CH_SA.load_state_dict(torch.load(f"{path_model_CH_SA}/{id_CH_SA}.pt", map_location=device))
model_CH_AB.load_state_dict(torch.load(f"{path_model_CH_AB}/{id_CH_AB}.pt", map_location=device))
model_AG_BC.load_state_dict(torch.load(f"{path_model_AG_BC}/{id_AG_BC}.pt", map_location=device))
model_AG_PA.load_state_dict(torch.load(f"{path_model_AG_PA}/{id_AG_PA}.pt", map_location=device))

model_CH_CA.to(device)
model_CH_PA.to(device)
model_CH_BC.to(device)
model_CH_EC.to(device)
model_CH_SA.to(device)
model_CH_AB.to(device)
model_AG_BC.to(device)
model_AG_PA.to(device)

'''
________________________________________________________________________________________________________________________
                                                    MENÚ
________________________________________________________________________________________________________________________
'''
while True:

    print("\nELIGE EL TIPO DE ANÁLISIS:\n\t" + "\x1b[1;32m" + " 1. Challenge Test\n\t 2. Aguas\n\t 3. PQ/PT\n\t "
                                                              "4. Ambientes\n\t 5. Superficies\n\t" + "\x1b[1;31m" + " 6. Salir" + '\033[0;m')

    menu1 = int(input("\n\tIntroduce tu opcion (1,2,3,4,5,6): "))

    if menu1 == 1:
        print("\nELIGE EL TIPO DE PATÓGENO:\n\t" + "\x1b[1;32m" + " 1. Aspergillus Brasiliensis\n\t "
                                                                  "2. Burkholderia Cepacia\n\t 3. Candida Albicans \n\t 4. Escherichia Coli\n\t 5. Pseudomonas Aeruginosa\n\t"
                                                                  " 6. Staphylococcus Aureus\n\t" + "\x1b[1;31m" + " 7. Volver al Menú ppal." + '\033[0;m')

        menu2 = int(input("\n\tIntroduce tu opcion (1,2,3,4,5,6,7): "))
        if menu2 == 7:
            print("\n___________________________________________")
            continue

    elif menu1 == 2:
        print(
            "\nELIGE EL TIPO DE PATÓGENO:\n\t" + "\x1b[1;32m" + " 1. Burkholderia Cepacia \n\t 2. Escherichia Coli\n\t 3. Pseudomonas Aeruginosa\n\t"
                                                                " 4. Aerobios Mesofilos\n\t 5. Mohos y levaduras\n\t" + "\x1b[1;31m" + " 6. Volver al Menú ppal." + '\033[0;m')

        menu2 = int(input("\n\tIntroduce tu opcion (1,2,3,4,5,6): "))
        if menu2 == 6:
            print("\n___________________________________________")
            continue

    elif menu1 == 3:
        print(
            "\nELIGE EL TIPO DE PATÓGENO:\n\t" + "\x1b[1;32m" + " 1. Burkholderia Cepacia \n\t 2. Candida Albicans\n\t 3. Escherichia Coli\n\t 4. Pseudomonas Aeruginosa\n\t"
                                                                " 5. Staphylococcus Aureus\n\t 6. Aerobios Mesofilos\n\t 7. Mohos y levaduras\n\t" + "\x1b[1;31m" + " 8. Volver al Menú ppal." + '\033[0;m')

        menu2 = int(input("\n\tIntroduce tu opcion (1,2,3,4,5,6,7,8): "))
        if menu2 == 8:
            print("\n___________________________________________")
            continue

    elif menu1 == 4 or menu1 == 5:
        print(
            "\nELIGE EL TIPO DE PATÓGENO:\n\t" + "\x1b[1;32m" + " 1. Aerobios Mesofilos \n\t 2. Mohos y levaduras\n\t" + "\x1b[1;31m" + " 3. Volver al Menú ppal." + '\033[0;m')

        menu2 = int(input("\n\tIntroduce tu opcion (1,2,3): "))
        if menu2 == 3:
            print("\n___________________________________________")
            continue

    elif menu1 == 6:
        quit()

    '''
    ____________________________________________________________________________________________________________________
                                    LECTURA DE IMAGENES SEGUN OPCION ESCOGIDA
    ____________________________________________________________________________________________________________________
    '''
    if menu1 == 1:
        deteccion = 1
        nombre_analisis = "Challenge Test"
        if menu2 == 1:
            filelist = glob.glob(f"./Imagenes/Challenge Test/Aspergillus Brasiliensis/*.png")
            nombre_patogeno = "Aspergillus Brasiliensis"
            model = model_CH_AB

        elif menu2 == 2:
            filelist = glob.glob(f"./Imagenes/Challenge Test/Burkholderia Cepacia/*.png")
            nombre_patogeno = "Burkholderia Cepacia"
            model = model_CH_BC

        elif menu2 == 3:
            filelist = glob.glob(f"./Imagenes/Challenge Test/Candida Albicans/*.png")
            nombre_patogeno = "Candida Albicans"
            model = model_CH_CA

        elif menu2 == 4:
            filelist = glob.glob(f"./Imagenes/Challenge Test/Escherichia Coli/*.png")
            nombre_patogeno = "Escherichia Coli"
            model = model_CH_EC

        elif menu2 == 5:
            filelist = glob.glob(f"./Imagenes/Challenge Test/Pseudomonas Aeruginosa/*.png")
            nombre_patogeno = "Pseudomonas Aeruginosa"
            model = model_CH_PA

        elif menu2 == 6:
            filelist = glob.glob(f"./Imagenes/Challenge Test/Staphylococcus Aureus/*.png")
            nombre_patogeno = "Staphylococcus Aureus"
            model = model_CH_SA

    elif menu1 == 2:
        nombre_analisis = "Aguas"
        if menu2 == 1:
            filelist = glob.glob(f"./Imagenes/Aguas/Burkholderia Cepacia/*.png")
            model = model_AG_BC
            deteccion = 0
            nombre_patogeno = "Burkholderia Cepacia"

        elif menu2 == 2:
            filelist = glob.glob(f"./Imagenes/Aguas/Escherichia Coli/*.png")
            deteccion = 0
            nombre_patogeno = "Escherichia Coli"

        elif menu2 == 3:
            filelist = glob.glob(f"./Imagenes/Aguas/Pseudomonas Aeruginosa/*.png")
            model = model_AG_PA
            deteccion = 0
            nombre_patogeno = "Pseudomonas Aeruginosa"

        elif menu2 == 4:
            filelist = glob.glob(f"./Imagenes/Aguas/Aerobios Mesofilos/*.png")
            deteccion = 1
            nombre_patogeno = "Aerobios Mesofilos"

        elif menu2 == 5:
            filelist = glob.glob(f"./Imagenes/Aguas/Mohos y levaduras/*.png")
            deteccion = 1

    elif menu1 == 3:
        nombre_analisis = "Superficies"
        deteccion = 1
        if menu2 == 1:
            filelist = glob.glob(f"./Imagenes/Superficies/Aerobios Mesofilos/*.png")
            nombre_patogeno = "Aerobios Mesofilos"

        elif menu2 == 2:
            filelist = glob.glob(f"./Imagenes/Superficies/Mohos y levaduras/*.png")
            nombre_patogeno = "Mohos y levaduras"

    elif menu1 == 3:
        nombre_analisis = "Producto Químico / Producto Terminado"
        deteccion = 0
        if menu2 == 1:
            filelist = glob.glob(f"./Imagenes/Producto Terminado/Burkholderia Cepacia/*.png")
            nombre_patogeno = "Burkholderia Cepacia"

        elif menu2 == 2:
            filelist = glob.glob(f"./Imagenes/Producto Terminado/Candida Albicans/*.png")
            nombre_patogeno = "Candida Albicans"

        elif menu2 == 3:
            filelist = glob.glob(f"./Imagenes/Producto Terminado/Escherichia Coli/*.png")
            nombre_patogeno = "Escherichia Coli"

        elif menu2 == 4:
            filelist = glob.glob(f"./Imagenes/Producto Terminado/Staphylococcus Aureus/*.png")
            nombre_patogeno = "Staphylococcus Aureus"

        elif menu2 == 5:
            filelist = glob.glob(f"./Imagenes/Producto Terminado/Pseudomonas Aeruginosa/*.png")
            nombre_patogeno = "Pseudomonas Aeruginosa"

        elif menu2 == 6:
            filelist = glob.glob(f"./Imagenes/Producto Terminado/Aerobios Mesofilos/*.png")
            nombre_patogeno = "Aerobios Mesofilos"
            deteccion = 1

        elif menu2 == 7:
            filelist = glob.glob(f"./Imagenes/Producto Terminado/Mohos y levaduras/*.png")
            nombre_patogeno = "Mohos y levaduras"
            deteccion = 1

    elif menu1 == 4:
        nombre_analisis = "Ambientes"
        deteccion = 1
        if menu2 == 1:
            filelist = glob.glob(f"./Imagenes/Ambientes/Aerobios Mesofilos/*.png")
            nombre_patogeno = "Aerobios Mesofilos"
            model = model_CH_CA

        elif menu2 == 2:
            filelist = glob.glob(f"./Imagenes/Ambientes/Mohos y levaduras/*.png")
            nombre_patogeno = "Mohos y levaduras"

    elif menu1 == 5:
        nombre_analisis = "Superficies"
        deteccion = 1
        if menu2 == 1:
            filelist = glob.glob(f"./Imagenes/Superficies/Aerobios Mesofilos/*.png")
            nombre_patogeno = "Aerobios Mesofilos"

        elif menu2 == 2:
            filelist = glob.glob(f"./Imagenes/Superficies/Mohos y levaduras/*.png")
            nombre_patogeno = "Mohos y levaduras"

    print(
        "\nHas elegido:\n\tTipo de analisis --> " + "\x1b[1;32m" + f"{nombre_analisis}" + '\033[0;m' + "\n\tTipo de patogeno --> " + "\x1b[1;32m" + f"{nombre_patogeno}\n" + '\033[0;m')
    '''
    ____________________________________________________________________________________________________________________
                                                INFERENCIA DETECCIÓN
    ____________________________________________________________________________________________________________________
    '''

    if deteccion == 1:
        inference_time_total = []
        boxes_pred_filt = []
        len_filelist = len(filelist)

        if len_filelist == 0:
            print("ALERTA --> No hay imagenes. Volviendo al menu principal...")
            continue

        print("Calculando predicciones...")

        # CALCULO DE PREDICCION
        for indice, img in enumerate(filelist):
            # Lectura de imagen
            img = plt.imread(img)

            # Transformacion a tensor
            img = T.to_tensor(img)

            # Inferencia del modelo
            model.eval()

            with torch.no_grad():
                start_time = time.time();
                prediction = (model([img.to(device)]));
                end_time = time.time();
                inference_time = (end_time - start_time);
                inference_time_total.append(inference_time)

            # Mostramos el tiempo de inferencia
            print("\tTiempo de inferencia (Imagen {} de {}): {:.4f} segundos".format(indice + 1, len_filelist,
                                                                                     inference_time))

            # Trabajamos las predicciones
            boxes_pred = prediction[0]["boxes"];
            boxes_pred = boxes_pred.cpu().numpy();

            # Aplicamos el Filtro
            boxes_pred_filt.append(
                filter_boxes(boxes_pred, iou_threshold=iou_threshold, dist_eucl_threshold=dist_eucl_threshold))

        inference_time_mean = sum(inference_time_total) / len(inference_time_total)
        print(f"\n\tPredicciones calculadas! --> Tiempo de inferencia medio: {inference_time_mean:.2f} segundos")

        # RESULTADOS
        print("\nMostrando resultados...")
        for indice, img in enumerate(filelist):
            # Obtenemos la cantidad de colonias predichas
            colonias_pred = len(boxes_pred_filt[indice])

            print(
                f"\tImagen {indice + 1}/{len_filelist} --> " + "\x1b[1;36m" + f"{colonias_pred} colonias" + '\033[0;m')

            fig = plt.figure()
            fig.canvas.set_window_title(f'Imagen {indice + 1}/{len_filelist}')
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], xticks=[], yticks=[])

            # Lectura de imagen
            image = plt.imread(img)

            ## Ploteamos predicciones e imagenes

            plt.imshow(image)

            ax.spines['left'].set_linewidth(0)
            ax.spines['right'].set_linewidth(0)
            ax.spines['bottom'].set_linewidth(0)
            ax.spines['top'].set_linewidth(0)
            plt.title(f"Nº Colonias: {colonias_pred}", color="black", fontsize=14, fontweight='bold')
            plt.suptitle(f"Tipo Analisis: {nombre_analisis}  -  Tipo patógeno: {nombre_patogeno}", fontsize=6)

            for i in boxes_pred_filt[indice]:
                xmax = int(i[2])
                ymax = int(i[3])
                xmin = int(i[0])
                ymin = int(i[1])
                width = xmax - xmin
                heigth = ymax - ymin
                rect = patches.Rectangle((xmin, ymin), width, heigth, edgecolor='r', facecolor='none', linewidth=1)
                ax.add_patch(rect)

            plt.show(block=False)
            plt.waitforbuttonpress(0)
            plt.close()

        '''
        ________________________________________________________________________________________________________________
                                                    INFERENCIA CLASIFICACION
        ________________________________________________________________________________________________________________
        '''
    else:
        preds = []
        inference_time_total = []
        len_filelist = len(filelist)

        if len_filelist == 0:
            print("ALERTA --> No hay imagenes. Volviendo al menu principal...")
            continue

        print("Calculando predicciones...")

        # CALCULO DE PREDICCION
        for indice, img in enumerate(filelist):
            # Lectura de imagen
            image = Image.open(img)

            # Transformaciones
            mean = np.array([0.5, 0.5, 0.5])
            std = np.array([0.5, 0.5, 0.5])
            loader = transforms.Compose(
                [transforms.Resize(128), transforms.ToTensor(), transforms.Normalize(std, mean)])
            image = loader(image).float()
            image = image.unsqueeze(0)
            image.to(device)

            # Inferencia
            model.eval()
            start_time = time.time()
            prediction = model(image)
            end_time = time.time();
            inference_time = (end_time - start_time);
            inference_time_total.append(inference_time)

            # Mostramos el tiempo de inferencia
            print("\tTiempo de inferencia Imagen ({} de {}): {:.4f} segundos".format(indice + 1, len_filelist,
                                                                                     inference_time))

            # Almacenamos resultadosb
            _, pred = torch.max(prediction, 1)
            pred = np.squeeze(pred.cpu().numpy())
            preds.append(pred)

        inference_time_mean = sum(inference_time_total) / len(inference_time_total)
        print(f"\n\tPredicciones calculadas! --> Tiempo de inferencia medio: {inference_time_mean:.2f} segundos")

        # RESULTADOS
        print("\nMostrando resultados...")
        for indice, img in enumerate(filelist):

            pred = preds[indice]

            # Convertimos prediccion
            if pred == 0:
                pred = " AUSENCIA "
                print(f"\tImagen {indice + 1}/{len_filelist} --> " + "\x1b[1;30;42m" + f"{pred}" + '\033[0;m')
                c = 'g'
            elif pred == 1:
                pred = " PRESENCIA "
                print(f"\tImagen {indice + 1}/{len_filelist} --> " + "\x1b[1;30;41m" + f"{pred}" + '\033[0;m')
                c = 'r'

            fig = plt.figure()
            fig.canvas.set_window_title(f'Imagen {indice + 1}/{len_filelist}')
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], xticks=[], yticks=[])

            # Lectura de imagen
            image = plt.imread(img)

            ## Ploteamos predicciones e imagenes
            plt.imshow(image)

            ax.spines['left'].set_linewidth(0)
            ax.spines['right'].set_linewidth(0)
            ax.spines['bottom'].set_linewidth(0)
            ax.spines['top'].set_linewidth(0)
            plt.title(f"{pred}", color=c, fontsize=14, fontweight='bold')
            plt.suptitle(f"Tipo Analisis: {nombre_analisis}  -  Tipo patógeno: {nombre_patogeno}\n", fontsize=6)

            plt.show(block=False)
            plt.waitforbuttonpress()
            plt.close()

    print("\nFIN. Volviendo al Menú principal...")
    print("\n___________________________________________")
