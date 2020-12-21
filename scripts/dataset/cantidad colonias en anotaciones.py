import xml.etree.cElementTree as ET
import os
import csv
import pandas as pd
import io
import cv2
import matplotlib.pyplot as plt
from matplotlib import patches


header = ["xmin", "ymin", "width", "heigth"]
mainpath_ann = f"./Annotations"

lista_df = os.listdir(mainpath_ann)
# print(lista_df)
cantidad_colonias_anotacion=[]

for i in lista_df:
    path_annotation_xml = f"{mainpath_ann}/{i}"
    annotation_xml = ET.parse(path_annotation_xml)
    root = annotation_xml.getroot()

    if (len(root))==1:
        j = root[0]
    else:
        j = root[1]

    data_string = j.text
    #for i in root:
        #data_string = j.text

    if data_string=="- ":
        cantidad_colonias_anotacion.append(0)
        continue

    data = io.StringIO(data_string.strip())
    df = pd.read_csv(data, sep=",", header=None, names=header, lineterminator=";")
    cantidad_colonias_anotacion.append(df.shape[0])
    # df.head()

print(sum(cantidad_colonias_anotacion))


print("Fin del analisis")