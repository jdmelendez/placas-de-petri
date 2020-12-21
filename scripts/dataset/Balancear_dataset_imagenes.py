
import os
import random

# LECTURA DE ARCHIVOS EN CARPETA

# Obtenemos la ruta desde donde se ejecuta nuestro script
dir_actual = os.getcwd()

# Buscamos las carpetas de carpetas
clases = os.listdir(dir_actual)


# El archivo de la lista de carpetas ".py", lo quitamos de la lista
clases = [x for x in clases if "." not in x ]
print("\n\nCLASES EXISTENTES:")
print(clases)

# Obtenemos las rutas de las distintas carpetas
paths = []
for i in range(len(clases)):
    paths.append(os.path.join(dir_actual, clases[i]))

# Vemos la cantidad de archivos de cada clase
print("\n\nCANTIDAD DE IMAGENES EN CADA CLASE:")
path_clases = []
for x in range(len(clases)):
    path_clases.append(os.listdir(paths[x]))
    print(f"Clase '{clases[x]}': {len(path_clases[x])} imágenes")


cantidad_minima_imagenes = 10000
cantidad_imagenes=[]
for i in range(len(path_clases)):
    cantidad_imagenes.append(len(path_clases[i]))
    if cantidad_imagenes[i] < cantidad_minima_imagenes:
        cantidad_minima_imagenes = cantidad_imagenes[i]

print("\nRealizando balance de imagenes...")

# ______________________________________________________________________________________________________________________
# ELIMINACION DE ARCHIVOS:

for i in range(len(clases)):
    if cantidad_imagenes[i] > cantidad_minima_imagenes:
        random.shuffle(path_clases[i])
        archivos_a_eliminar = path_clases[i][cantidad_minima_imagenes:]
        for file in archivos_a_eliminar:
            os.remove(os.path.join(paths[i], file))

# ______________________________________________________________________________________________________________________
# VERIFICACION DATASET BALANCEADO:

print("\n\nCANTIDAD DE IMAGENES EN CADA CLASE:")
path_clases = []
for x in range(len(clases)):
    path_clases.append(os.listdir(paths[x]))
    print(f"Clase '{clases[x]}': {len(path_clases[x])} imágenes")










