"""
A traves de este script se realizan las siguientes funciones:

    1. Lectura del dataset-original, copia de los ficheros, y division en 'train' y 'test' dentro
       de una carpeta llamada 'ToDrive'. Para dividir el conjunto se ha de tener en cuenta que en los ficheros
       de test, se tenga el blister formado por la imagen de las dos camaras.   
    2. Aumento de la cantidad de imagenes del conjunto de 'train' y de 'test' mediante volteos y simetrias. 

    # pyinstaller -c --onefile --name Deteccion-Dataset ./scripts/dataset/deteccion-main-dataset.py


"""

from subprocess import Popen, call
import time


call('python ../scripts/dataset/deteccion-train-test-dataset.py')
call('python ../scripts/dataset/deteccion-aumentar-dataset.py')
