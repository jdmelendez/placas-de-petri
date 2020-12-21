import time
start = time.time()

# ______________________________________________________________________________________________________________________
# LIBRERIAS

import torch
import torch.nn.functional as F
import numpy as np
from torchvision import datasets, transforms
from torch import nn
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
import seaborn as sns
import math
import cv2

# ______________________________________________________________________________________________________________________
# CLASES

clases = ["AUSENCIA", "PRESENCIA"] # TODO
n_clases = len(clases)


# ______________________________________________________________________________________________________________________
# ARQUITECTURA DE LA RED NEURONAL

# Definimos la red neuronal convolucional:
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # Capa convolucional de entrada (profundidad imagen entrada, profundidad imagen
        self.conv1 = nn.Conv2d(3, 128, 3, padding=1)  # TODO
        # Capa convolucional --> 400 x 400 x 64
        self.conv2 = nn.Conv2d(128, 256, 3, padding=1)  # TODO
        # Capa convolucional --> 200 x 200 x 128
        self.conv3 = nn.Conv2d(256, 512, 3, padding=1)  # TODO
        # Max pooling (Kernel, stride)
        self.pool = nn.MaxPool2d(2, 2)  # TODO
        # Capa fully conected
        self.fc1 = nn.Linear(512 * 16 * 16, 1000)  # TODO
        # Capa fully conected salida (tamaño, numero de carpetas)
        self.fc2 = nn.Linear(1000, n_clases)  # TODO
        # Abandono --> Evitar overfitting
        self.dropout = nn.Dropout(0.25)  # TODO

    def forward(self, x):

        # Secuencia de capas convolucionales y max pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        # Se aplana la matriz --> Vector
        x = x.view(-1, 512 * 16 * 16)  # TODO
        # Capa de abandono
        x = self.dropout(x)
        # Capa oculta con Relu
        x = F.relu(self.fc1(x))
        # Capa de abandono
        x = self.dropout(x)
        # Capa oculta de salida con Relu
        x = self.fc2(x)
        return x


model=Net()
print("\nARQUITECTURA DE LA RED:")
print(model)


# ______________________________________________________________________________________________________________________
# DISPONIBILIDAD GPU/CPU

# Comprobamos si la GPU esta disponible:
train_on_gpu = torch.cuda.is_available()
print("\nDISPOSITIVO DE ENTRENAMIENTO:")
if not train_on_gpu:
    print("CUDA no esta disponible. Trabajando en CPU...\n\n")
else:
    print("CUDA disponible! Trabajando en GPU...\n\n")

# Movemos la red a CUDA si esta disponible:
if train_on_gpu:
   model.cuda()

# ______________________________________________________________________________________________________________________
#  CARGANDO EL MODELO DE LA RED

model.load_state_dict(torch.load(f"./Red_entrenada.pt",map_location=torch.device('cpu')))

# ______________________________________________________________________________________________________________________
# CARGA DE IMAGENES

# Transformaciones a los datos de entrada
mean = np.array([0.5, 0.5, 0.5])  # TODO
std = np.array([0.5, 0.5, 0.5])  # TODO

data_transforms = transforms.Compose([
    transforms.Resize(128),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

batch_size_test=2
# Adquisicion de imagen testeada
data_dir_test = "./Imagenes/test"
test_data = datasets.ImageFolder(data_dir_test, transform=data_transforms)
num_workers = 0  # Numero de subprocesos paralelos al cargar los datos TODO
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size_test )


# ______________________________________________________________________________________________________________________
# EVALUACION:

# Inicializamos variables:
test_loss = 0.0
class_correct = list(0. for i in range(n_clases))
class_total = list(0. for i in range(n_clases))
total = 0.0
correct_total = 0.0
confusion_matrix = torch.zeros(n_clases, n_clases)
target_v, preds_v, data_v, prob_v = [], [], [],[]

model.eval()

# Iteramos con el conjunto de datos de test:
for data, target in test_loader:
    # Movemos los tensores a la GPU si esta disponible:
    if train_on_gpu:
        data, target = data.cuda(), target.cuda()

    # Forward pass: Se obtienen las predicciones del modelo:
    output = model(data)

    # Convertimos las probabilidades a las carpetas predichas:
    _, pred = torch.max(output, 1)
    preds = np.squeeze(pred.numpy()) if not train_on_gpu else np.squeeze(pred.cpu().numpy())

    # Obtenemos los scores de cada imagen
    sm = torch.nn.Softmax()
    prob = sm(output)
    prob, _ = torch.max(prob, 1)
    prob = np.squeeze(prob.detach().numpy()) if not train_on_gpu else np.squeeze(prob.cpu().detach().numpy())
    prob = prob * 100
    np.set_printoptions(precision=2)
    np.set_printoptions(suppress=True)

    # Comparamos predicciones por clase con lo verdadero:
    correct = (pred == target).squeeze()

    # Calculo precision por clase:
    for i in range(batch_size_test):
        label = target[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1

    # Almacenamos en vectores los datos para plotearlos despues
    data = data.cpu()
    for i in range(batch_size_test):
        target_v.append(clases[target[i]])
        preds_v.append(clases[preds[i]])
        data_v.append(data[i])
        prob_v.append(prob[i])


        # Matriz de confusion
    for t, p in zip(target.view(-1), pred.view(-1)):
        confusion_matrix[t.long(), p.long()] += 1


# ______________________________________________________________________________________________________________________
# PLOT DE RESULTADOS:

# plot confussion matrix
plt.figure(figsize=(8,4))
heatmap=sns.heatmap(confusion_matrix, xticklabels=clases, yticklabels=clases, annot=True, fmt=".0f",cmap="YlGn",annot_kws={"size": 10});
heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0)


plt.title("Confusion matrix",fontsize=10,fontweight='bold')
plt.ylabel('True class',fontsize=10)
plt.xlabel('Predicted class',fontsize=10)
plt.tick_params(labelsize=10)
plt.tight_layout()
plt.show()
cv2.waitKey(0)


# Mostramos la precision por carpetas y total:
print("\n\nRECALL/ACCURACY POR CLASES:\n")
for i in range(n_clases):
    print("Recall of %5s: %.2f %%  (%2d/%2d)" %(clases[i],100*class_correct[i]/class_total[i],np.sum(class_correct[i]),np.sum(class_total[i])))


print("Accuracy of %5s: %.2f %%  (%2d/%2d)" %(clases[0],100*class_correct[0]/(class_correct[0]+(class_total[1]- class_correct[1])),np.sum(class_correct[0]), class_correct[0]+class_total[1]- class_correct[1]))
print("Accuracy of %5s: %.2f %%  (%2d/%2d)" %(clases[1],100*class_correct[1]/(class_correct[1]+(class_total[0]- class_correct[0])),np.sum(class_correct[1]), class_correct[1]+class_total[0]- class_correct[0]))

print(f"\nTotal Accuracy: %.2f %% (%2d/%2d)" %(100 *np.sum(class_correct)/np.sum(class_total) ,np.sum(class_correct),np.sum(class_total)))



# ______________________________________________________________________________________________________________________
# PLOT DE LAS IMAGENES FALLIDAS:

def imshow(img):
  img=img/2+0.5
  npimg=img.numpy()
  plt.imshow(np.transpose(npimg,(1,2,0)))
  plt.show()

images_failed = []
preds_failed = []
probs_failed = []

for i in range(len(test_data)):
    if preds_v[i] != target_v[i]:
        images_failed.append(data_v[i])
        preds_failed.append(preds_v[i])
        probs_failed.append(prob_v[i])

    else:
        continue

n_images_failed = len(images_failed)
print(f"Imágenes fallidas: {n_images_failed}\n")

fig = plt.figure(figsize=(10,  n_images_failed))
for i in range(n_images_failed):
    ax = fig.add_subplot(int(math.ceil(n_images_failed / 3)), 3, 1 + i, xticks=[], yticks=[])
    imshow(images_failed[i])
    ax.set_title("{} - {:.2f} %".format(preds_failed[i], probs_failed[i]), color="red", fontsize=15, fontweight='bold')
    plt.tight_layout()
    ax.spines['left'].set_linewidth(3)
    ax.spines['right'].set_linewidth(3)
    ax.spines['bottom'].set_linewidth(3)
    ax.spines['top'].set_linewidth(3)
    ax.spines['right'].set_color("red")
    ax.spines['left'].set_color("red")
    ax.spines['bottom'].set_color("red")
    ax.spines['top'].set_color("red")


# ______________________________________________________________________________________________________________________
# PLOT DE TODAS LAS IMAGENES :

#
#
# fig = plt.figure(figsize=(135, 135))
# for i in range(len(test_data)) :
#     ax = fig.add_subplot(220,6,1+i,xticks=[], yticks=[])
#     imshow(data_v[i])
#     ax.set_title("{}".format(preds_v[i]),
#                  color=("green" if preds_v[i]==target_v[i] else "red"),fontsize=3,fontweight='bold')
#     ax.spines['left'].set_linewidth(3)
#     ax.spines['right'].set_linewidth(3)
#     ax.spines['bottom'].set_linewidth(3)
#     ax.spines['top'].set_linewidth(3)
#     if preds_v[i]==target_v[i]:
#       ax.spines['right'].set_color("green")
#       ax.spines['left'].set_color("green")
#       ax.spines['bottom'].set_color("green")
#       ax.spines['top'].set_color("green")
#     else:
#       ax.spines['right'].set_color("red")
#       ax.spines['left'].set_color("red")
#       ax.spines['bottom'].set_color("red")
#       ax.spines['top'].set_color("red")