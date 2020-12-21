

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
from torchvision import datasets, transforms
from torch import nn, random

import numpy as np
import math
import time
import glob
import matplotlib.pyplot as plt
from matplotlib import patches
import os
import msvcrt


'''
________________________________________________________________________________________________________________________
                                       DEFINICION MODELO CLASIFICACION
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

model_AG_BC = Net()

'''
________________________________________________________________________________________________________________________
                                             CARGA MODELOS
________________________________________________________________________________________________________________________
'''
print("Cargando modelos...")


id_AG_BC= 340

path_models = "C:/Users/jmelendez/OneDrive - RNBCOSMETICOS/Escritorio/Proyecto_Placas de Petri/Modelos Entrenados"
path_model_AG_BC = os.path.join(path_models,"Aguas/Burkholderia Cepacia")

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu');
model_AG_BC.load_state_dict(torch.load(f"{path_model_AG_BC}/{id_AG_BC}-AG-BC.pt",map_location=device))

# COMPROBAMOS SI EXISTE GPU Y MOVEMOS EL MODELO AL DISPOSITIVO:

model_AG_BC.to(device)

'''
________________________________________________________________________________________________________________________
                                            INFERENCIA CLASIFICACION
________________________________________________________________________________________________________________________
'''
filelist = glob.glob(f"./Aguas/Burkholderia Cepacia/*.png")
model = model_AG_BC

preds=[]
from PIL import Image
# CALCULO DE PREDICCION
for indice,img in enumerate(filelist):

  # Lectura de imagen
  image = Image.open(img)

  # Transformaciones
  mean = np.array([0.5, 0.5, 0.5])  # TODO
  std = np.array([0.5, 0.5, 0.5])  # TODO
  loader = transforms.Compose([transforms.Resize(128), transforms.ToTensor(), transforms.Normalize(std, mean)])
  image = loader(image).float()
  image = image.unsqueeze(0)
  image.to(device)




  model.eval()
  prediction=model(image)
  _, pred = torch.max(prediction, 1)
  pred = np.squeeze(pred.cpu().numpy())

  if pred==0:
    pred = "AUSENCIA"
  elif pred ==1:
    pred = "PRESENCIA"

  preds.append(pred)

# RESULTADOS
for indice, img in enumerate(filelist):
  fig = plt.figure()
  ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], xticks=[], yticks=[])

  # Lectura de imagen
  image = plt.imread(img)

  pred=preds[indice]


  ## Ploteamos predicciones e imagenes

  plt.imshow(image)

  ax.spines['left'].set_linewidth(0)
  ax.spines['right'].set_linewidth(0)
  ax.spines['bottom'].set_linewidth(0)
  ax.spines['top'].set_linewidth(0)
  plt.title(f"{pred}", color="black", fontsize=10, fontweight='bold')

  plt.show(block=True)





  #with torch.no_grad():
    #prediction = (model([image.to(device)]))
#   _, pred = torch.max(output, 1)
#   preds = np.squeeze(pred.cpu().numpy())
#
#   # Obtenemos los scores de cada imagen
#   sm = torch.nn.Softmax(dim=1)
#   prob = sm(output)
#   prob, _ = torch.max(prob, 1)
#   prob = np.squeeze(prob.cpu().detach().numpy())
#   prob = prob * 100















