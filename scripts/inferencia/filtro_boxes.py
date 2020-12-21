import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patches
iou_threshold = 0.3
dist_eucl_threshold=30
boxes = [[ 862.4010,  333.2581, 1010.8207,  487.2140],
           [ 659.1379,  539.6116,  773.2083,  661.0372],
           [ 261.5770,  803.9460,  365.7910,  909.4618],
           [ 407.4993,  685.8335,  440.7177,  718.9224],
           [ 410.1616,  482.8670,  443.3765,  515.5571],
           [ 622.2744,  583.2274,  655.4390,  616.3936],
           [ 635.4080, 1000.2378,  668.6801, 1033.2041],
           [ 863.9162,  457.7025,  897.5240,  490.6759],
           [1092.7468,  372.8358, 1125.0078,  405.4874],
           [ 195.9000,  958.3295,  228.8799,  991.1713],
           [ 728.6602,  622.4073,  762.1271,  655.0460],
           [ 720.2173,  617.6186,  755.0497,  650.5895],
           [ 736.6530,  619.5341,  769.3837,  649.2589],
           [ 621.6802,  926.7149,  653.7390,  958.0774],
           [ 298.6200,  844.2845,  329.9874,  877.0437],
           [ 615.7368,  589.8966,  650.4445,  624.0845],
           [ 606.2358,  619.4561,  638.0149,  650.9217],
           [ 616.9898,  975.6024,  690.6036, 1048.4250],
           [ 929.2977,  397.5111,  963.4476,  431.2442],
           [ 385.2245,  458.6689,  460.2112,  531.5613],
           [ 382.8869,  665.4819,  456.7999,  736.2513],
           [ 699.4494,  582.8334,  732.4670,  617.0742],
           [ 613.8248,  921.7125,  646.4430,  953.4778],
           [ 731.3521,  718.5121,  763.7970,  749.4279],
           [ 602.5086,  564.9429,  668.5520,  634.7711],
           [ 844.5164,  441.4579,  913.4290,  510.2438],
           [1077.8297,  353.0391, 1142.7025,  422.5533],
           [ 806.3895,  758.3198,  838.3345,  788.2294]]

print(len(boxes))

boxes=np.asarray(boxes)
df1=pd.DataFrame(data=boxes,columns=["x1","y1","x2","y2"])
print(df1)
image = plt.imread("45.png")
fig = plt.figure()
ax = fig.add_axes([0,0,1,1],xticks=[],yticks=[])
plt.imshow(image)



for index, row in df1.iterrows():

    xmin =int(row['x1'])
    ymin =int(row['y1'])
    xmax = int(row['x2'])
    ymax = int(row['y2'])

    rect = patches.Rectangle((xmin, ymin), xmax-xmin,ymax-ymin, edgecolor='r', facecolor='none', linewidth=1.5)
    ax.annotate(f"{index}",(xmin,ymin),color='b')
    ax.add_patch(rect)


plt.show(block=False)
#def filter_boxes(boxes, iou_threshold=0.1, dist_eucl_threshold=0.1):
"""
Calculate the Intersection over Union (IoU) of two bounding boxes.

Parameters
----------
bb1 : dict
Keys: {'x1', 'x2', 'y1', 'y2'}
The (x1, y1) position is at the top left corner,
the (x2, y2) position is at the bottom right corner
bb2 : dict
Keys: {'x1', 'x2', 'y1', 'y2'}
The (x, y) position is at the top left corner,
the (x2, y2) position is at the bottom right corner

Returns
-------
float
in [0, 1]
"""

boxes_filt = boxes
deletes = 0
indices_delete = []
indice_i=0


for i in boxes:
    x1A = i[0]
    y1A = i[1]
    x2A = i[2]
    y2A = i[3]

    indice_j = 0
    for j in boxes[indice_i:]:

        iou = 0
        dist_eucl = 1000
        x1B = j[0]
        y1B = j[1]
        x2B = j[2]
        y2B = j[3]

        # Arriba izquierda - Abajo derecha
        if ((x1A < x1B) and (y1A < y1B) and (x2A < x2B) and (y2A < y2B) and (x1B < x2A) and (y1B < y2A)) or ((x1A > x1B) and (y1A > y1B) and (x2A > x2B) and (y2A > y2B) and (x1A < x2B) and (y1A < y2B)):
            x_left = max(x1A, x1B)
            y_top = max(y1A, y1B)
            x_right = min(x2A, x2B)
            y_bottom = min(y2A, y2B)

        # Arriba derecha
        elif ((x1A < x1B) and (y1A > y1B) and (x2A < x2B) and (y2A > y2B) and (x1B < x2A) and (y1A < y2B)):
            x_left = x1B
            y_top = y1A
            x_right = x2A
            y_bottom = y2B

        # Abajo izquierda
        elif ((x1A > x1B) and (y1A < y1B) and (x2A > x2B) and (y2A < y2B) and (x1A < x2B) and (y1B < y2A)):
            x_left = x1A
            y_top = y1B
            x_right = x2B
            y_bottom = y2A

        # Arriba
        elif ((x1A < x1B) and (y1A > y1B) and (x2A > x2B) and (y2A > y2B) and (y1A < y2B) and (x1B < x2A)):
            x_left = x1B
            y_top = y1A
            x_right = x2B
            y_bottom = y2B

        # Derecha
        elif ((x1A < x1B) and (y1A < y1B) and (x2A < x2B) and (y2A > y2B) and (y1A < y2B) and (x1B < x2A)):
            x_left = x1B
            y_top = y1B
            x_right = x2A
            y_bottom = y2B

        # Abajo
        elif ((x1A < x1B) and (y1A < y1B) and (x2A > x2B) and (y2A < y2B) and (y1A < y2B) and (x1B < x2A)):
            x_left = x1B
            y_top = y1B
            x_right = x2B
            y_bottom = y2A

        # Izquierda
        elif ((x1A > x1B) and (y1A < y1B) and (x2A > x2B) and (y2A > y2B) and (y1A < y2B) and (x1B < x2A)):
            x_left = x1A
            y_top = y1B
            x_right = x2B
            y_bottom = y2B


        # Arriba ancho
        elif ((x1A > x1B) and (y1A > y1B) and (x2A < x2B) and (y2A > y2B) and (y1A > y2B) and (x1A < x2B)):
            x_left = x1A
            y_top = y1A
            x_right = x2A
            y_bottom = y2B

        # Bajo ancho
        elif ((x1A > x1B) and (y1A < y1B) and (x2A < x2B) and (y2A < y2B) and (y1A < y2B) and (x1A < x2B)):
            x_left = x1A
            y_top = y1B
            x_right = x2A
            y_bottom = y2A

        # Izquierda ancho
        elif ((x1A > x1B) and (y1A > y1B) and (x2A > x2B) and (y2A < y2B) and (y1A < y2B) and (x1A < x2B)):
            x_left = x1A
            y_top = y1A
            x_right = x2B
            y_bottom = y2A

        # Derecha ancho
        elif ((x1A < x1B) and (y1A > y1B) and (x2A < x2B) and (y2A < y2B) and (y1A < y2B) and (x1A < x2B)):
            x_left = x1B
            y_top = y1A
            x_right = x2A
            y_bottom = y2A

        # Dentro
        elif (x1A < x1B and y1A < y1B and x2A > x2B and y2A > y2B) or (
        (x1A > x1B and y1A > y1B and x2A < x2B and y2A < y2B)):
            centroAx = x1A+(x2A - x1A) / 2
            centroAy = y1A+(y2A - y1A) / 2
            centroBx = x1B+(x2B - x1B) / 2
            centroBy = y1B+(y2B - y1B) / 2
            dist_eucl = math.sqrt((centroAx - centroBx) ** 2 + (centroAy - centroBy) ** 2)


            if dist_eucl < dist_eucl_threshold:
                boxes_filt[indice_i + indice_j][:] = [0, 0, 0, 0]

            indice_j += 1
            continue

        else:
            indice_j+=1
            continue

        # The intersection of two axis-aligned bounding boxes
        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # compute the area of both AABBs
        bb1_area = (x2A - x1A) * (y2A - y1A)
        bb2_area = (x2B - x1B) * (y2B - y1B)

        # compute the intersection over union
        iou = intersection_area / float(bb1_area + bb2_area - intersection_area)

        if (iou > iou_threshold):
            boxes_filt[indice_i+indice_j][:] = [0,0,0,0]

        indice_j+=1
    indice_i+=1


for i in range(len(boxes_filt)):
    result = np.all((boxes_filt[i]==0))
    if result:
        indices_delete.append(i)

boxes_filt=np.delete(boxes_filt,indices_delete,axis=0)


print(len(boxes_filt))
#return boxes_filt

df=pd.DataFrame(data=boxes_filt,columns=["x1","y1","x2","y2"])
print(df)

image = plt.imread("Inv_48-10.png")
fig = plt.figure()
ax = fig.add_axes([0,0,1,1],xticks=[],yticks=[])
plt.imshow(image)



for index, row in df.iterrows():

    xmin =int(row['x1'])
    ymin =int(row['y1'])
    xmax = int(row['x2'])
    ymax = int(row['y2'])

    rect = patches.Rectangle((xmin, ymin), xmax-xmin,ymax-ymin, edgecolor='r', facecolor='none', linewidth=1.5)
    ax.annotate(f"{index}",(xmin,ymin),color='b')
    ax.add_patch(rect)


plt.show()

