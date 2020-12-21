from PyQt5 import QtCore
from GUI_pipeline_inferencia import pipeline_inferencia, predecir


class CalculaPrediccion(QtCore.QThread):
    envio_prediccion = QtCore.pyqtSignal(object)
    envio_valor_progressBar = QtCore.pyqtSignal(int)

    def __init__(self, path,  parent=None):
        QtCore.QThread.__init__(self, parent)
        self.path = path
        # self.barra = barra

    def run(self):

        try:
            modelos, device, id_imagenes_lista, paths_imagenes, imagenes_lista = pipeline_inferencia(
                self.path)

            predicciones = {paths_imagenes[indice]: [predecir(
                modelos=modelos, imagen=imagen, device=device, indice=indice, id_imagenes_lista=id_imagenes_lista), self.envio_valor_progressBar.emit((indice+1)*100/len(id_imagenes_lista))]
                for indice, imagen in enumerate(imagenes_lista)}

            self.envio_prediccion.emit(predicciones)

        except:
            self.envio_prediccion.emit(
                {"Error": "Todas las placas han sido ya evaluadas"})
            return

    def stop(self):
        self.terminate()
