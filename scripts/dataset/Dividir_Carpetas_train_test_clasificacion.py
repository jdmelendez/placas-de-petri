'''

ESTE SCRIPT DEBE EJECUTARSE EN UN DIRECTORIO QUE CONTENGA LAS DISTINTAS CLASES

'''

train_split = 0.9  # TODO Tamaño del dataset de "training" (sobre 1)
    
# LIBRERIAS:

import os
import numpy as np
import random
import shutil


# Obtenemos la ruta desde donde se ejecuta nuestro script
dir_actual = os.getcwd()

# Buscamos las carpetas de carpetas
clases = os.listdir(dir_actual)

# El archivo de la lista de carpetas ".py", lo quitamos de la lista
clases = [x for x in clases if "." not in x ]
print("\n\nCLASES EXISTENTES:")
print(clases)

# Creacion de carpetas:
train_dir = os.path.join(dir_actual, 'train')
test_dir = os.path.join(dir_actual, 'test')

train_dir_clases=[]
test_dir_clases=[]
for i in range(len(clases)):
    train_dir_clases.append(os.path.join(train_dir, clases[i]))
    test_dir_clases.append(os.path.join(test_dir, clases[i]))

try:
    print("\n" + "Creando carpetas...")
    # Train:
    os.mkdir(train_dir)
    os.mkdir(test_dir)

    for i in range(len(test_dir_clases)):
        os.mkdir(train_dir_clases[i])
        os.mkdir(test_dir_clases[i])

except FileExistsError:
    print('Las carpetas ya han sido creadas.' + "\n")

dir_clases=[]
for i in range(len(clases)):
    dir_clases.append(os.path.join(dir_actual, clases[i]))


# Recopilación de archivos de carpetas originales:
files_clases=[]
for i in range(len(clases)):
    files_clases.append([os.path.join(dir_clases[i], f) for f in os.listdir(dir_clases[i]) if os.path.isfile(os.path.join(dir_clases[i], f))])

# Split de separación de training
msk_clases=[]
for i in range(len(clases)):
    for j in range(len(files_clases[i])):
        msk_clases.append(np.random.rand(len(files_clases[i])) < train_split)

rand_items_clases=[]
for i in range(len(clases)):
    rand_items_clases.append(random.sample(files_clases[i], int(len(files_clases[i]) * train_split)) )


# Funcion para mover imagenes desde carpetas origen a "train" y "valid"
def move_file_list(directory, file_list):
    for f in file_list:
        f_name = f.split('/')[-1]
        shutil.move(f, directory)


# LLamada a la función:
for i in range(len(clases)):
    move_file_list(train_dir_clases[i],rand_items_clases[i])

files_clases_test=[]
# Imagenes que quedan en la carpeta original para test:
for i in range(len(clases)):
    files_clases_test.append([os.path.join(dir_clases[i], f)
                  for f in os.listdir(dir_clases[i])
                  if os.path.isfile(os.path.join(dir_clases[i], f))])


# LLamada a la función:
for i in range(len(clases)):
    move_file_list(test_dir_clases[i], files_clases_test[i])

# Eliminacion de carpetas vacias:
for i in dir_clases:
    os.rmdir(i)

print("Division en TRAIN y TEST...FINALIZADA!")
