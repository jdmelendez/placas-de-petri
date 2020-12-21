'''

ESTE SCRIPT DEBE EJECUTARSE EN UN DIRECTORIO QUE CONTENGA LAS DISTINTAS CLASES

'''

import glob
import shutil
import random
import os
train_split = 0.9  # TODO Tamaño del dataset de "training" (sobre 1)
seed = 5  # TODO Semilla de aletoriedad

# LIBRERIAS:


# Obtenemos la ruta desde donde se aloja el dataset-original
dir_original = './datasets/dataset-deteccion-original'
dir_deteccion = './datasets/dataset-deteccion'

# Elegir tipo de analisis
tipos_analisis = os.listdir(dir_original)

print("\nELIGE TIPO DE ANALISIS:")
for indice, tipo in enumerate(tipos_analisis):
    print(f"{indice+1} --> {tipo}")

menu1 = int(input("\nIntroduce tu opcion (numero):"))
tipo_analisis_elegido = tipos_analisis[menu1 - 1]
dir_tipo_analisis_elegido = os.path.join(dir_original, tipo_analisis_elegido)


# Elegir tipo de patogenos
tipos_patogenos = os.listdir(dir_tipo_analisis_elegido)

print("\nELIGE TIPO DE PATOGENO:")
for indice, tipo in enumerate(tipos_patogenos):
    print(f"{indice+1} --> {tipo}")

menu2 = int(input("\nIntroduce tu opcion (numero):"))
tipo_patogeno_elegido = tipos_patogenos[menu2 - 1]
dir_tipo_patogeno_elegido = os.path.join(
    dir_tipo_analisis_elegido, tipo_patogeno_elegido)
dir_original = dir_tipo_patogeno_elegido


# Vaciamos el contenido del dataset-de-detecion
for the_file in os.listdir(dir_deteccion):
    file_path = os.path.join(dir_deteccion, the_file)
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print(e)


# Copiamos el dataset original en el la carpeta de dataset-deteccion para trabajar con el
carpetas_dir_original = os.listdir(dir_original)
for carpeta in carpetas_dir_original:
    try:
        os.mkdir(os.path.join(dir_deteccion, carpeta))
    except FileExistsError:
        print(f'La carpeta {carpeta} ya existe.' + "\n")

    ficheros_en_carpeta_original = os.listdir(
        os.path.join(dir_original, carpeta))

    for fichero in ficheros_en_carpeta_original:
        shutil.copy(os.path.join(
            dir_original, f"{carpeta}/{fichero}"), os.path.join(dir_deteccion, carpeta))


# Buscamos las carpetas de carpetas
carpetas = os.listdir(dir_original)

# Creamos la carpeta donde se almacenara el conjunto de entrenamiento y test
try:
    print("\nCreando carpeta ToDrive...")

    os.mkdir(os.path.join(dir_deteccion, 'ToDrive'))

except FileExistsError:
    print('Las carpeta ToDrive ya estaba creada.' + "\n")

# El archivo de la lista de carpetas ".py", lo quitamos de la lista
carpetas = [x for x in carpetas if "." not in x]
print("\n\nCARPETAS EXISTENTES:")
print(carpetas)

# Creacion de carpetas:
train_dir = os.path.join(dir_deteccion, 'ToDrive/train')
test_dir = os.path.join(dir_deteccion, 'ToDrive/test')

train_dir_carpetas = []
test_dir_carpetas = []

for i in range(len(carpetas)):
    train_dir_carpetas.append(os.path.join(train_dir, carpetas[i]))
    test_dir_carpetas.append(os.path.join(test_dir, carpetas[i]))

try:
    print("\n" + "Creando carpetas...")
    # Train:
    os.mkdir(train_dir)
    os.mkdir(test_dir)

    for i in range(len(test_dir_carpetas)):
        os.mkdir(train_dir_carpetas[i])
        os.mkdir(test_dir_carpetas[i])

except FileExistsError:
    print('Las carpetas ya existen.' + "\n")

dir_carpetas = []
for i in range(len(carpetas)):
    dir_carpetas.append(os.path.join(dir_deteccion, carpetas[i]))


# Recopilación de archivos de carpetas originales:
files_carpetas = []
for i in range(len(carpetas)):
    files_carpetas.append([os.path.join(dir_carpetas[i], f) for f in os.listdir(
        dir_carpetas[i]) if os.path.isfile(os.path.join(dir_carpetas[i], f))])


# Split de separación de training
nombres_archivos = []
nombres_archivos_random = []
for i in range(len(carpetas)):
    nombres_archivos.append(os.listdir(dir_carpetas[i]))
    random.seed(seed)
    nombres_archivos_random.append(random.sample(
        nombres_archivos[i], k=int(len(nombres_archivos[i])*train_split)))

lista_annotations = []
lista_images = []
# Añadir nombres arhivos random a la ruta correspondiente:

for j in range(len(nombres_archivos_random[0])):
    lista_annotations.append(os.path.join(
        dir_carpetas[0], nombres_archivos_random[0][j]))

for j in range(len(nombres_archivos_random[1])):
    lista_images.append(os.path.join(
        dir_carpetas[1], nombres_archivos_random[1][j]))

# Funcion para mover imagenes desde carpetas origen a "train" y "test"


def move_file_list(directory, file_list):
    for f in file_list:
        f_name = f.split('/')[-1]
        shutil.move(f, directory)


#
# LLamada a la función:
move_file_list(train_dir_carpetas[0], lista_annotations)
move_file_list(train_dir_carpetas[1], lista_images)

files_clases_test = []

# Imagenes que quedan en la carpeta original para test:
for i in range(len(carpetas)):
    files_clases_test.append([os.path.join(dir_carpetas[i], f)
                              for f in os.listdir(dir_carpetas[i])
                              if os.path.isfile(os.path.join(dir_carpetas[i], f))])


# LLamada a la función:
for i in range(len(carpetas)):
    move_file_list(test_dir_carpetas[i], files_clases_test[i])

# Eliminacion de carpetas vacias:
for i in dir_carpetas:
    os.rmdir(i)

print("Division en TRAIN y TEST...FINALIZADA!")
