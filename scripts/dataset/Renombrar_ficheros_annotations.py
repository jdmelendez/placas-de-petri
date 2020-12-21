'''

ESTE SCRIPT SE HA DE EJECUTAR FUERA DE LA CARPETA DE "ANNOTATIONS"

'''

import os


# Obtenemos la ruta desde donde se ejecuta nuestro script:
dir_actual = os.getcwd()

# Obtenemos los nombres archivos actuales:
path=f"{dir_actual}/Annotations/"
nombres_ficheros_actuales = os.listdir(path)

# Renombramos los archivos .XML para quitarles el Identificador (se elimina "XX_"):
nombres_ficheros_nuevos=[]
for nombre_fichero in nombres_ficheros_actuales:
    for i in nombre_fichero:
        if i != "_":
            nombre_fichero = nombre_fichero[1:]
        elif i == "_":
            nombres_ficheros_nuevos.append(nombre_fichero[1:])
            break

# Recorremos el directorio y cambiamos los nombres actuales por lo nuevos:
for i,nombre_archivo_actual in enumerate(os.listdir(path)):
    dst = nombres_ficheros_nuevos[i]
    src = path + nombre_archivo_actual
    dst = path + dst
    os.rename(src,dst)

print("\nFicheros '.xml' renombrados correctamente!")

