import xml.etree.cElementTree as ET
import os
import csv
import pandas as pd
import io
import cv2
import matplotlib.pyplot as plt
from matplotlib import patches


header = ["xmin", "ymin", "width", "heigth"]
# mainpath_ann = f"./Annotations"
mainpath_ann = f"./datasets/dataset-deteccion/Ambientes/Mohos y levaduras/Annotations"

lista_df = os.listdir(mainpath_ann)
# print(lista_df)


for i in lista_df:
    path_annotation_xml = f"{mainpath_ann}/{i}"
    annotation_xml = ET.parse(path_annotation_xml)
    root = annotation_xml.getroot()

    if (len(root)) == 1:
        j = root[0]
    else:
        j = root[0]
        if j.text == '- ':
            j = root[1]
            if j.text == '- ':
                continue

    data_string = j.text

    # for j in root:
    #     data_string = j.text
    #
    # if data_string == "- ":
    #     continue

    data = io.StringIO(data_string.strip())
    df = pd.read_csv(data, sep=",", header=None,
                     names=header, lineterminator=";")
    # df.head()

    # data_filterx = df[df["xmin"] <= 5]
    # if data_filterx.empty:
    #     pass
    # else:
    #     print(f"\n{i}")
    #     print(data_filterx)

    # data_filtery = df[df["ymin"] <= 5]
    # if data_filtery.empty:
    #     pass
    # else:
    #     print(f"\n{i}")
    #     print(data_filtery)

    data_filterh = df[df["heigth"] <= 5]
    if data_filterh.empty:
        pass
    else:
        print(f"\n{i}")
        print(data_filterh)

    data_filterw = df[df["width"] <= 5]
    if data_filterw.empty:
        pass
    else:
        print(f"\n{i}")
        print(data_filterw)


print("Fin del analisis")
