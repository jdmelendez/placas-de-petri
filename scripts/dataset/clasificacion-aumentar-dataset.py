'''
    SE TIENE QUE EJECUTAR ESTE SCRIPT EN LA CARPETA QUE CONTIENE LAS DISTINTAS CLASES

'''

import cv2
import os
import glob



# ______________________________________________________________________________________________________________________
# FUNCION ROTACION
def rotacion(imagen, angulo, centro=None, escala=1.0):

    # Obtenemos ancho y alto
    ancho = imagen.shape[1]
    alto = imagen.shape[0]

    if centro is None:
        centro = (ancho // 2, alto // 2)

    # Matriz de transformación, rotación
    M = cv2.getRotationMatrix2D(centro, angulo, escala)

    # Aplicamos rotación a la imagen
    ImagenRotada = cv2.warpAffine(imagen, M, (ancho, alto))

    # Devolvemos la imagen rotada
    return ImagenRotada


# ______________________________________________________________________________________________________________________
# FUNCION INVERTIR
def invertir(imagen):

    ImagenInvertida = cv2.flip(imagen, 0)

    return ImagenInvertida

# ______________________________________________________________________________________________________________________
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
for x in range(len(clases)):
    cantidad_imagenes = os.listdir(paths[x])
    print(f"Clase '{clases[x]}': {len(cantidad_imagenes)} imágenes")

# ______________________________________________________________________________________________________________________
# APLICAMOS LA ROTACION:

# Recorremos cada clase y aplicamos la transformacion sobre cada imagen:
for x in range(len(clases)):
    imagenes = [cv2.imread(file) for file in glob.glob(f"{paths[x]}/*.png")]
    print(f"\n\nTransformando imagenes de la clase {clases[x]}...")

    for y in range(len(imagenes)):

        nombres_imagenes = os.listdir(paths[x])
        nombre_imagen = nombres_imagenes[y]
        
        ImagenRotada90 = rotacion(imagenes[y], 90, None, 1.0)
        ImagenRotada180 = rotacion(imagenes[y], 180, None, 1.0)
        ImagenRotada270 = rotacion(imagenes[y], 270, None, 1.0)
        ImagenInvertida = invertir(imagenes[y])
        ImagenRotada90Inv = rotacion(ImagenInvertida, 90, None, 1.0)
        ImagenRotada180Inv = rotacion(ImagenInvertida, 180, None, 1.0)
        ImagenRotada270Inv = rotacion(ImagenInvertida, 270, None, 1.0)
        cv2.imwrite(f"{paths[x]}/r90_{nombre_imagen}", ImagenRotada90)
        cv2.imwrite(f"{paths[x]}/r180_{nombre_imagen}", ImagenRotada180)
        cv2.imwrite(f"{paths[x]}/r270_{nombre_imagen}", ImagenRotada270)
        cv2.imwrite(f"{paths[x]}/Inv_{nombre_imagen}", ImagenInvertida)
        cv2.imwrite(f"{paths[x]}/Inv_r90_{nombre_imagen}", ImagenRotada90Inv)
        cv2.imwrite(f"{paths[x]}/Inv_r180_{nombre_imagen}", ImagenRotada180Inv)
        cv2.imwrite(f"{paths[x]}/Inv_r270_{nombre_imagen}", ImagenRotada270Inv)

for x in range(len(clases)):
    cantidad_imagenes = os.listdir(paths[x])
    print(f"Clase '{clases[x]}': {len(cantidad_imagenes)} imágenes")
















