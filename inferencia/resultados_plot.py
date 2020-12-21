import matplotlib.pyplot as plt
from matplotlib import patches
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def resultados_plot(paths_imagenes, predicciones):

    dict_prediccion_e_imagenes = dict(zip(paths_imagenes, predicciones))

    for path_imagen, prediccion in dict_prediccion_e_imagenes.items():

        ax = genera_figura(path_imagen=path_imagen)

        # if prediccion is clases:
        if isinstance(prediccion, str):
            resultados_clasificacion_plot(prediccion=prediccion)
        else:
            resultados_deteccion_plot(boxes=prediccion, ax=ax)

        plt.show(block=False)
        plt.waitforbuttonpress(0)
        plt.close()


def resultados_deteccion_plot(boxes, ax):

    plt.title(f"Nº Colonias: {len(boxes)}",
              color="black", fontsize=14, fontweight='bold')

    for i in boxes:
        xmax = int(i[2])
        ymax = int(i[3])
        xmin = int(i[0])
        ymin = int(i[1])
        width = xmax - xmin
        heigth = ymax - ymin
        rect = patches.Rectangle(
            (xmin, ymin), width, heigth, edgecolor='r', facecolor='none', linewidth=1)
        ax.add_patch(rect)


def resultados_clasificacion_plot(prediccion):

    plt.title(f"Clase: {prediccion}",
              color="black", fontsize=14, fontweight='bold')


def genera_figura(path_imagen):

    fig = plt.figure()
    fig.canvas.set_window_title('Imagen')
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], xticks=[], yticks=[])
    imagen = plt.imread(path_imagen)
    plt.imshow(imagen)
    ax.spines['left'].set_linewidth(0)
    ax.spines['right'].set_linewidth(0)
    ax.spines['bottom'].set_linewidth(0)
    ax.spines['top'].set_linewidth(0)

    return ax
