import math
from matplotlib import patches
import matplotlib.pyplot as plt
import matplotlib
import xml.etree.cElementTree as ET
import pandas as pd
import numpy as np
import csv
import io
import glob
import os
import cv2
nombre = "Inv_r90_121"

'''
r90_    
r180_
r270_
Inv_
Inv_r90_
Inv_r180_
Inv_r270_
'''


path_annotation_xml = f"C:/Users/jmelendez/GIT/placas-de-petri/laboratory/Annotations/{nombre}.xml"
annotation_xml = ET.parse(path_annotation_xml)
root = annotation_xml.getroot()

header = ["xmin", "ymin", "width", "heigth"]

for i in root:
    data_string = i.text
    if data_string != '- ':
        break


data = io.StringIO(data_string.strip())
df = pd.read_csv(data, sep=",", header=None, names=header, lineterminator=";")
print(df)

image = plt.imread(
    f"C:/Users/jmelendez/GIT/placas-de-petri/laboratory/Images/{nombre}.png")
fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1], xticks=[], yticks=[])
plt.imshow(image)


for index, row in df.iterrows():

    xmin = int(row['xmin'])
    ymin = int(row['ymin'])
    width = int(row['width'])
    heigth = int(row['heigth'])

    rect = patches.Rectangle((xmin, ymin), width, heigth,
                             edgecolor='r', facecolor='none', linewidth=1.5)
    # ax.annotate(f"{index}",(xmin,ymin),color='b')
    ax.add_patch(rect)


plt.show()
