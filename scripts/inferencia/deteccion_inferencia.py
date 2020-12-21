
"""DETECCION_Inferencia.ipynb
    https://colab.research.google.com/drive/15NZH_nJyEKLzARTUPDBOYlY_d10pU93p
"""

'''
________________________________________________________________________________________________________________________
                                                    PARÁMETROS
________________________________________________________________________________________________________________________
'''

# Parametros filtro
iou_threshold=0.3
dist_eucl_threshold = 20

id = "119"
id_analisis="AG"
id_patogeno = "AM"

'''
________________________________________________________________________________________________________________________
                                                  LIBRERIAS
________________________________________________________________________________________________________________________
'''

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F

import numpy as np
import math
import time
import glob
import matplotlib.pyplot as plt
from matplotlib import patches



'''
________________________________________________________________________________________________________________________
                                                DEFINICION MODELO
________________________________________________________________________________________________________________________
'''


model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

# # Cantidad de clases que tenemos (backgound + colonia)
num_classes = 2

# # Obtenemos el numero de caracteristicas de entrada al clasificador
in_features = model.roi_heads.box_predictor.cls_score.in_features

# # Reemplazamos la cabecera de la red pre-trained por la nueva
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# # Cantidad maxima de objetos a detectar
model.roi_heads.detections_per_img=400

# COMPROBAMOS SI EXISTE GPU Y MOVEMOS EL MODELO AL DISPOSITIVO:
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu');

'''
________________________________________________________________________________________________________________________
                                                CARGA MODELO
________________________________________________________________________________________________________________________
'''

model.load_state_dict(torch.load(f"../Inferencia/deteccion/{id}-{id_analisis}-{id_patogeno}.pt",map_location=device))
model.to(device)

'''
________________________________________________________________________________________________________________________
                                                    FILTRO
________________________________________________________________________________________________________________________
'''

def filter_boxes(boxes,iou_threshold, dist_eucl_threshold):
    
  boxes_filt = boxes
  deletes = 0
  indices_delete = []
  indice_i=0


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
      if ((x1A < x1B) and (y1A < y1B) and (x2A < x2B) and (y2A < y2B) and (x1B < x2A) and (y1B < y2A)) or ((x1A > x1B) and (y1A > y1B) and (x2A > x2B) and (y2A > y2B) and (x1A < x2B) and (y1A < y2B)):
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
      elif ((x1A > x1B) and (y1A< y1B) and (x2A < x2B) and (y2A < y2B) and (y1A < y2B) and (x1A < x2B)):
        x_left = x1A
        y_top = y1B
        x_right = x2A
        y_bottom = y2A

      # Izquierda ancho
      elif ((x1A > x1B) and (y1A> y1B) and (x2A > x2B) and (y2A < y2B) and (y1A < y2B) and (x1A < x2B)):
        x_left = x1A
        y_top = y1A
        x_right = x2B
        y_bottom = y2A

      # Derecha ancho
      elif ((x1A < x1B) and (y1A> y1B) and (x2A < x2B) and (y2A < y2B) and (y1A < y2B) and (x1A < x2B)):
        x_left = x1B
        y_top = y1A
        x_right = x2A
        y_bottom = y2A

      # Horizontal
      elif ((x1A > x1B) and (y1A< y1B) and (x2A < x2B) and (y2A > y2B) and (y1A < y2B) and (x1A < x2B)):
        x_left = x1A
        y_top = y1B
        x_right = x2A
        y_bottom = y2B

      # Vertical
      elif ((x1A < x1B) and (y1A> y1B) and (x2A > x2B) and (y2A < y2B) and (y1A < y2B) and (x1A < x2B)):
        x_left = x1B
        y_top = y1A
        x_right = x2B
        y_bottom = y2A

      
      # Dentro
      elif (x1A < x1B and y1A < y1B and x2A > x2B and y2A > y2B) or ((x1A > x1B and y1A > y1B and x2A < x2B and y2A < y2B)):
        centroAx = x1A+(x2A - x1A) / 2
        centroAy = y1A+(y2A - y1A) / 2
        centroBx = x1B+(x2B - x1B) / 2
        centroBy = y1B+(y2B - y1B) / 2
        dist_eucl = math.sqrt((centroAx - centroBx)**2 + (centroAy - centroBy)**2)

        if dist_eucl < dist_eucl_threshold:
          boxes_filt[indice_i + indice_j][:] = [0, 0, 0, 0]
        indice_j += 1
        continue
      
      else:
        indice_j+=1
        continue

      # The intersection of two axis-aligned bounding boxes
      intersection_area = (x_right - x_left) * (y_bottom - y_top)

      # compute the area of both AABBs
      bb1_area = (x2A - x1A) * (y2A - y1A)
      bb2_area = (x2B - x1B) * (y2B - y1B)

      # compute the intersection over union
      iou = intersection_area / float(bb1_area + bb2_area - intersection_area)

      if (iou > iou_threshold):
          boxes_filt[indice_i+indice_j][:] = [0,0,0,0]

      indice_j+=1
    indice_i+=1


  for i in range(len(boxes_filt)):
    result = np.all((boxes_filt[i]==0))
    if result:
      indices_delete.append(i)

  boxes_filt=np.delete(boxes_filt,indices_delete,axis=0)

  return boxes_filt

'''
________________________________________________________________________________________________________________________
                                                  INFERENCIA
________________________________________________________________________________________________________________________
'''


## LECTURA DE LAS IMÁGENES:
filelist = glob.glob(f"../Inferencia/deteccion/Images/*.png")

boxes_pred_filt=[]

for indice,img in enumerate(filelist):

  # Lectura de imagen
  image = plt.imread(img)

  # Transformacion a tensor
  img = F.to_tensor(image)

  # Inferencia del modelo
  model.eval();
  with torch.no_grad():
    start_time = time.time();
    prediction = (model([img.to(device)]));
    end_time = time.time();
    inference_time = (end_time - start_time);

  # Mostramos el tiempo de inferencia
  print("\nTiempo de inferencia: {:.4f} segundos".format(inference_time))

  # Trabajamos las predicciones
  boxes_pred = prediction[0]["boxes"];
  boxes_pred = boxes_pred.cpu().numpy();

  # Aplicamos el Filtro
  boxes_pred_filt.append(filter_boxes(boxes_pred, iou_threshold=iou_threshold, dist_eucl_threshold=dist_eucl_threshold))
  # if len(boxes_pred_filt[indice]) == 1:
  #   xmax = boxes_pred_filt[0][2]
  #   xmin = boxes_pred_filt[0][0]
  #   if xmax - xmin < 5:
  #     boxes_pred_filt = []

'''
________________________________________________________________________________________________________________________
                                                  RESULTADOS
________________________________________________________________________________________________________________________
'''


for indice,img in enumerate(filelist):

  fig = plt.figure()
  ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], xticks=[], yticks=[])

  # Lectura de imagen
  image = plt.imread(img)
  # Obtenemos la cantidad de colonias predichas
  colonias_pred = len(boxes_pred_filt[indice])

  ## Ploteamos predicciones e imagenes

  plt.imshow(image)

  ax.spines['left'].set_linewidth(0)
  ax.spines['right'].set_linewidth(0)
  ax.spines['bottom'].set_linewidth(0)
  ax.spines['top'].set_linewidth(0)
  plt.title(f"Nº Colonias: {colonias_pred}",color="black",fontsize=10,fontweight='bold')

  for i in boxes_pred_filt[indice]:
    xmax =int(i[2])
    ymax =int(i[3])
    xmin =int(i[0])
    ymin =int(i[1])
    width = xmax-xmin
    heigth = ymax-ymin
    rect = patches.Rectangle((xmin,ymin), width, heigth, edgecolor ='r', facecolor = 'none',linewidth=1)
    ax.add_patch(rect)

  plt.show(block=True)





