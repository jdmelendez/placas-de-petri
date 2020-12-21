# ================================================= LIBRERIAS ======================================================
import sys
import os
from PyQt5 import QtCore, QtGui, uic, QtWidgets
from PyQt5.QtWidgets import QApplication, QListWidgetItem, QAction, QMessageBox, QFileSystemModel, QColumnView, QTableWidgetItem
from PyQt5.QtCore import QDir, QRect
from PyQt5.QtGui import QColor, QImage, QPixmap, QPen, QPainter, QFont, QStandardItemModel, QBrush
from PyQt5.QtSql import QSqlDatabase, QSqlQueryModel, QSqlQuery
from pathlib import Path
import glob
import shutil
import qdarkstyle

from GUI_config import SERVER_NAME, DATABASE_NAME, USERNAME, PASSWORD, CONN_STRING, PATH_IMGS_PRUEBA, TEXTO_BOTONES_PATOGENO, TEXTO_CARPETAS_ANALISIS, TEXTO_BOTONES_DILUCION, TEXTO_BOTONES_TIEMPO, TEXTO_COMBOBOX_DILUCION, TEXTO_COMBOBOX_ZONAS
from GUI_lista_directorios import lista_primer_directorio, lista_resto_directorios
from GUI_QThread_Prediccion import CalculaPrediccion
from GUI_QThread_Camara import ActivaCamara
from GUI_guarda_ImagenFichero_prediccion import guarda_ImagenFichero_prediccion
from GUI_crea_mensaje_alerta import crea_mensaje_alerta
from GUI_muestra_resultado_enLabel import muestra_resultado_enLabel
from GUI_comprueba_click_dentro_imagen import comprueba_click_dentro_imagen
from GUI_dibuja_region import dibuja_region
from GUI_valida_cambios_edicion import valida_cambios_edicion
from GUI_borrar_region import borrar_region


class Main(QtWidgets.QMainWindow):
    def __init__(self):
        super(Main, self).__init__()

 # ================================================= CARGA FICHERO UI ==================================================

        uifile = os.path.join(os.path.abspath(
            os.path.dirname(__file__)), 'InterfaceUI.ui')
        uic.loadUi(uifile, self)

 # ================================================= INICIA BOTONES ====================================================

        # Boton añadir carpeta:
        self.boton_anadir_carpeta.clicked.connect(self.fn_boton_anadir_carpeta)

        # Boton borrar fichero
        self.boton_borrar_fichero.clicked.connect(self.fn_boton_borrar_fichero)

        # Boton reset
        self.boton_Reset.clicked.connect(self.fn_boton_Reset)

        # Botones patogeno
        self.boton_AB.clicked.connect(
            lambda: self.obten_nombre_patogeno_analisis(nombre_boton=self.boton_AB))
        self.boton_BC.clicked.connect(
            lambda: self.obten_nombre_patogeno_analisis(nombre_boton=self.boton_BC))
        self.boton_CA.clicked.connect(
            lambda: self.obten_nombre_patogeno_analisis(nombre_boton=self.boton_CA))
        self.boton_EC.clicked.connect(
            lambda: self.obten_nombre_patogeno_analisis(nombre_boton=self.boton_EC))
        self.boton_PA.clicked.connect(
            lambda: self.obten_nombre_patogeno_analisis(nombre_boton=self.boton_PA))
        self.boton_SA.clicked.connect(
            lambda: self.obten_nombre_patogeno_analisis(nombre_boton=self.boton_SA))
        self.boton_MO.clicked.connect(
            lambda: self.obten_nombre_patogeno_analisis(nombre_boton=self.boton_MO))
        self.boton_LE.clicked.connect(
            lambda: self.obten_nombre_patogeno_analisis(nombre_boton=self.boton_LE))
        self.boton_AM.clicked.connect(
            lambda: self.obten_nombre_patogeno_analisis(nombre_boton=self.boton_AM))

        # Botones dilucion
        self.boton_D1.clicked.connect(
            lambda: self.obten_nombre_dilucion(nombre_boton=self.boton_D1))
        self.boton_D2.clicked.connect(
            lambda: self.obten_nombre_dilucion(nombre_boton=self.boton_D2))
        self.boton_D3.clicked.connect(
            lambda: self.obten_nombre_dilucion(nombre_boton=self.boton_D3))
        self.comboBox_DOtro.currentIndexChanged.connect(
            self.obten_nombre_dilucion_comboBox)

        # Botones tiempo
        self.boton_T0.clicked.connect(
            lambda: self.obten_nombre_tiempo(nombre_boton=self.boton_T0))
        self.boton_T7.clicked.connect(
            lambda: self.obten_nombre_tiempo(nombre_boton=self.boton_T7))
        self.boton_T14.clicked.connect(
            lambda: self.obten_nombre_tiempo(nombre_boton=self.boton_T14))
        self.boton_T28.clicked.connect(
            lambda: self.obten_nombre_tiempo(nombre_boton=self.boton_T28))

        # Boton zona
        self.comboBox_zona.currentIndexChanged.connect(
            self.obten_nombre_zona_comboBox)

        # Botones ver-ocultar
        self.boton_ver.clicked.connect(self.fn_boton_ver)
        self.boton_ocultar.clicked.connect(self.fn_boton_ocultar)
        # Boton inferencia
        self.boton_Evaluar.clicked.connect(self.fn_boton_Evaluar)

        # Botones ventana
        finish = QAction("Quit", self)
        finish.triggered.connect(self.closeEvent)

        # Tabla resultados
        # columnas = ['ID Placa', 'CH', 'Ref', 'Resultado']
        # self.tabla.setColumnCount(4)
        # self.tabla.setRowCount(1)
        # self.tabla.setHorizontalHeaderLabels(columnas)
        # self.tabla.setItem(0, 0, QTableWidgetItem('Hola'))
        # self.tabla.cellClicked.connect(self.fn_mostrarItem)

        self.db = QSqlDatabase.addDatabase('QODBC')
        self.db.setDatabaseName(CONN_STRING)
        self.db.open()
        self.qry = QSqlQuery(self.db)
        self.modelo_query = QSqlQueryModel()

        SQL_STATEMENT = 'INSERT INTO dbo.Tabla_Placa(id_TipoAnalisis, id_TipoPatogeno) VALUES (AG, EC)'
        self.ejecuta_query(SQL_STATEMENT)

        # Botones camara
        self.boton_camara.clicked.connect(self.fn_boton_camara)
        self.boton_CapturaImagen.clicked.connect(self.fn_boton_CapturaImagen)
        self.FLAG_CAMARA_ACTIVA = 0
        self.FLAG_CAPTURA_IMAGEN = 0
        self.inicializa_nombre_imagen()

        # Botones edicion
        self.boton_lapiz.clicked.connect(self.fn_boton_lapiz)
        self.boton_borrar.clicked.connect(self.fn_boton_borrar)
        self.boton_validar.clicked.connect(self.fn_boton_validar)
        self.boton_cancelar.clicked.connect(self.fn_boton_cancelar)
        self.boton_ausencia.clicked.connect(self.fn_boton_ausencia)
        self.boton_presencia.clicked.connect(self.fn_boton_presencia)
        self.visor.mousePressEvent = self.mousePressEvent
        self.primer_click_x = 0
        self.primer_click_y = 0
        self.segundo_click_x = 0
        self.segundo_click_y = 0
        self.lista_nuevas_boxes = []
        self.lista_borrar_boxes = []

        self.FLAG_DIBUJAR = False
        self.FLAG_BORRAR = False
        self.FLAG_PRIMER_CLICK = True
        self.FLAG_SEGUNDO_CLICK = False
        self.FLAG_VALIDAR_EDICION_DETECCION = False
        self.FLAG_VALIDAR_EDICION_CLASIFICACION = False

 # ================================================= INICIA DIRECTORIOS ============================================
        """ El pipeline de este apartado se encarga de obtener y mostrar las imagenes que hay en el directorio, incluyendo
        las distintas subcarpetas de tipos de analisis, refererencias, etc. Concretamente, al clicar sobre el directorio
        de tipos de analisis, se actualiza su hijo, y asi consecutivamente. Para ello se han creado unas funciones externas
        que se encargan de ello. Además, tambien se obtiene la ruta de la imagen objetivo seleccionada para poder usarla
        en el resto de funciones de la interfaz. Si el elemento seleccionado es una imagen, se mostrará en el visor. Si
        esta imagen se ha inferenciado anteriormente, se mostrará la imagen con las regiones. Si no se ha inferenciado,
        se habilitará el botón de evaluacion de placa.
        """
        # Asociamos la ruta donde se encuentran las carpetas de los tipos de analisis y creamos el objeto para listar el directorio
        self.dirModel = QFileSystemModel()
        self.dirModel.setRootPath(PATH_IMGS_PRUEBA)

        # Mediante la funcion externa, se recorren todos los ficheros existentes en el directorio padre
        lista_primer_directorio(self.lista_analisis, self.dirModel)

        # Mediante los clicks en las carpetas de los subdirectorios se llama a ciertas funciones que recorren el interior de
        # la carpeta seleccionada y lo muestran por pantalla.
        self.lista_analisis.clicked.connect(
            lambda: lista_resto_directorios(self.lista_analisis, self.lista_lotes, self.lista_imagenes, self.dirModel, 1))
        self.lista_lotes.clicked.connect(lambda: lista_resto_directorios(
            self.lista_lotes, self.lista_imagenes, None, self.dirModel))
        self.lista_lotes.clicked.connect(lambda:
                                         self.obten_path_del_directorio(Qlist=self.lista_lotes))
        self.lista_imagenes.clicked.connect(lambda:
                                            self.obten_path_del_directorio(Qlist=self.lista_imagenes))

    # La siguiente funcion obtiene el elemento seleccionado de los directorios, y en caso de ser una imagen, la mostrará en el visor.
    # tambien se adquiere el nombre del elemento seleccionado para poder trabajar futuramente con el.

    def obten_path_del_directorio(self, Qlist):
        index = Qlist.currentIndex()
        self.path_seleccionado_del_directorio = self.dirModel.filePath(index)
        self.elemento_seleccionado = self.path_seleccionado_del_directorio.split(
            '/')[-1]
        self.label_elemento_seleccionado.setText(self.elemento_seleccionado)
        self.boton_Evaluar.setEnabled(True)

        if self.path_seleccionado_del_directorio[-3:] == 'png':
            self.decide_que_imagen_colocar_en_visor()

 # ================================================= VISOR DE IMAGEN ==================================================
        """Este apartado contiene las funciones que manejan la imagen que se coloca en el visor. En funcion de si la imagen
        ha sido inferenciada o no, se carga su solucion o no. Si existe solucion, se permite activar el modo edicion, y
        los botones de ver/ocultar regiones. Si se ha clicado una imagen y anteriormente la camara estaba encendida, esta
        se apagara, y la imagen mostrada se sustituira por la seleccionada.
        """

    def decide_que_imagen_colocar_en_visor(self):
        # Miramos si la imagen ha sido inferencia, y en caso de que si, la colocamos en el visor. Si no, colocamos la imagen
        # sin inferenciar.
        self.path_seleccionado_del_directorio_modificado = self.path_seleccionado_del_directorio

        self.lista_archivos_directorio = os.listdir(os.path.join(
            self.path_seleccionado_del_directorio_modificado, os.pardir))
        imagen_con_OK = self.path_seleccionado_del_directorio_modificado.split(
            '/')[-1][:-4] + "_OK.png"

        self.activar_desactivar_botones_edicion(0)
        self.boton_Evaluar.setEnabled(True)
        self.label_resultado_prediccion.setText("Resultado no disponible")

        if imagen_con_OK in self.lista_archivos_directorio:
            self.path_seleccionado_del_directorio_modificado = self.path_seleccionado_del_directorio_modificado[
                : -4] + "_OK.png"

            self.boton_ver.setDisabled(True)
            self.boton_ocultar.setEnabled(True)
            self.activar_desactivar_botones_edicion(1)
            self.label_resultado_prediccion.setText(muestra_resultado_enLabel(
                self.path_seleccionado_del_directorio_modificado))

        if self.FLAG_CAMARA_ACTIVA:
            self.activa_desactiva_camara(1)

        self.coloca_imagen_seleccionada_en_visor(
            path=self.path_seleccionado_del_directorio_modificado)

    def coloca_imagen_seleccionada_en_visor(self, path, camara=0, imagen_camara=None):

        # Colocamos la imagen correspondiente en el visor
        if camara == 1:
            imagen_visor = QPixmap.fromImage(imagen_camara)
        else:
            imagen_visor = QPixmap(f"{path}")

        self.imagen_visor_escalada = imagen_visor.scaled(int(self.visor.width()), int(
            self.visor.height()), QtCore.Qt.KeepAspectRatio)
        self.visor.setPixmap(self.imagen_visor_escalada)

        # Obtenemos las medidas del visor y de la imagen para despues poder comprobar si los click en modo edicion se efectuan dentro de la imagen
        ancho_imagen_visor = imagen_visor.width()
        alto_imagen_visor = imagen_visor.height()
        ancho_imagen_visor_escalada = self.imagen_visor_escalada.width()
        alto_imagen_visor_escalada = self.imagen_visor_escalada.height()
        ancho_visor = self.visor.width()
        alto_visor = self.visor.height()

        self.ancho_zonaClick_imagen = [
            (ancho_visor-ancho_imagen_visor_escalada)/2, (ancho_visor-ancho_imagen_visor_escalada)/2+ancho_imagen_visor_escalada]
        self.alto_zonaClick_imagen = [
            (alto_visor-alto_imagen_visor_escalada)/2, (alto_visor-alto_imagen_visor_escalada)/2+alto_imagen_visor_escalada]
        self.offset_X = (ancho_visor - ancho_imagen_visor_escalada)/2
        self.offset_Y = (alto_visor - alto_imagen_visor_escalada) / 2
        self.factor_escala_dimension_x = ancho_imagen_visor_escalada/ancho_imagen_visor
        self.factor_escala_dimension_y = alto_imagen_visor_escalada/alto_imagen_visor

 # ================================================== MANEJAR CAMARA ==================================================
        """A traves del siguiente apartado se maneja mediante multihilo la ejecucion de la camara, de manera que esta se activa
        o desactiva segun los botones pulsados. En funcion de esto, se activan y desactivan ciertos botones. Ademas,
        al final del apartado se configura el estilo del boton de activar la camara para saber la sitaucion de esta.
        """

    def fn_boton_camara(self):
        self.activa_desactiva_camara(self.FLAG_CAMARA_ACTIVA)

    def recibeImagenCamara(self, imagen_camara):
        self.coloca_imagen_seleccionada_en_visor(
            path=None, camara=1, imagen_camara=imagen_camara)
        # self.lista_imagenes.setDisabled(False)

        if self.FLAG_CAPTURA_IMAGEN:
            self.guarda_imagen_capturada(imagen_camara)

    def activa_desactiva_camara(self, FLAG_CAMARA):

        # Se enchufa la camara
        if FLAG_CAMARA == 0:

            self.camaraQThread = ActivaCamara()
            self.boton_camara.setDisabled(True)
            self.camaraQThread.envio_imagen.connect(self.recibeImagenCamara)
            self.camaraQThread.envio_imagen_capturar.connect(
                self.recibeImagenCamara)
            self.camaraQThread.start()
            self.boton_CapturaImagen.setEnabled(True)
            self.activar_desactivar_botones_eleccion(1)
            self.activar_desactivar_botones_edicion(0)
            self.styleSheet_boton_camara(FLAG_CAMARA)
            self.boton_camara.setDisabled(False)
            self.FLAG_CAMARA_ACTIVA = 1

        # Se apaga la camara
        else:
            self.camaraQThread.stop()
            self.boton_camara.setDisabled(True)
            self.boton_CapturaImagen.setDisabled(True)
            self.activar_desactivar_botones_eleccion(0)
            self.styleSheet_boton_camara(FLAG_CAMARA)
            self.boton_camara.setDisabled(False)
            self.FLAG_CAMARA_ACTIVA = 0

    def fn_boton_CapturaImagen(self):
        self.FLAG_CAPTURA_IMAGEN = 1

    def styleSheet_boton_camara(self, FLAG_CAMARA):
        if FLAG_CAMARA:
            self.boton_camara.setStyleSheet("QPushButton{background-color: #06aa58;; border: None; border-radius: 5px;}"
                                            "QPushButton:disabled{background-color: rgb(124, 124, 124);} #gris oscuro")
            self.boton_camara.setText("ACTIVAR CAMARA")
        else:
            self.boton_camara.setStyleSheet(
                "QPushButton{background-color: #06aa58;; border: None; border-radius: 5px;}"
                "QPushButton:disabled{background-color: rgb(124, 124, 124);} #gris oscuro")
            self.boton_camara.setText("DESACTIVAR CAMARA")

 # ============================================== GUARDAR IMAGEN CAMARA ===============================================
        """A traves del siguiente apartado se configuran las funciones que permiten guardar una imagen cuando se pulsa el
        boton de capturar imagen. Simplemente, se forma el nombre de la imagen en funcion de los botones seleccionados (tipo
        de patogeno, referencia, fecha, etc), y este nombre se une a la ruta donde se quiere guardar la imagen,
        definida por las carpetas seleccionadas en el menu de archivos. En caso de que el nombre de imagen ya existe,
        automaticamente se renombran las imagenes para que no se sobreescriban.
        """

    def guarda_imagen_capturada(self, imagen_camara):
        imagen = QPixmap.fromImage(imagen_camara)
        carpeta_destino = self.path_seleccionado_del_directorio
        self.comprobar_si_existe_nombre(self.nombre_imagen)
        imagen.save(f"{carpeta_destino}/{self.nombre_imagen}.png")
        self.FLAG_CAPTURA_IMAGEN = 0

    def obten_nombre_patogeno_analisis(self, nombre_boton):
        try:
            self.id_analisis = TEXTO_CARPETAS_ANALISIS[self.path_seleccionado_del_directorio.split(
                '/')[-2]]
        except:
            texto = "Debes de seleccionar antes una carpeta de referencia (COLUMNA CENTRAL)"
            crea_mensaje_alerta(texto)

        self.id_num = ""
        self.id_patogeno = f"_{TEXTO_BOTONES_PATOGENO[nombre_boton.text()]}"
        self.forma_nombre_imagen()

    def obten_nombre_dilucion(self, nombre_boton):
        self.id_dilucion = f"-{TEXTO_BOTONES_DILUCION[nombre_boton.text()]}"
        self.forma_nombre_imagen()

    def obten_nombre_dilucion_comboBox(self):
        self.id_dilucion = f"-{TEXTO_COMBOBOX_DILUCION[self.comboBox_DOtro.currentText()]}"
        self.forma_nombre_imagen()

    def obten_nombre_tiempo(self, nombre_boton):
        self.id_tiempo = f"-{TEXTO_BOTONES_TIEMPO[nombre_boton.text()]}"
        self.forma_nombre_imagen()

    def obten_nombre_zona_comboBox(self):
        self.id_zona = f"-{TEXTO_COMBOBOX_ZONAS[self.comboBox_zona.currentText()]}"
        self.forma_nombre_imagen()

    def forma_nombre_imagen(self):
        self.nombre_imagen = f"{self.id_analisis}{self.id_patogeno}{self.id_num}{self.id_tiempo}{self.id_dilucion}{self.id_zona}"

        # self.comprobar_si_existe_nombre(self.nombre_imagen)

        self.label_nombre_imagen.setText(self.nombre_imagen)

    def comprobar_si_existe_nombre(self, nombre_imagen):
        if (f"{nombre_imagen}.png" or f"{nombre_imagen[0:5]}0{nombre_imagen[5:]}.png") in os.listdir(self.path_seleccionado_del_directorio):
            id = nombre_imagen.split('-')[0]
            if len(id) == 6:
                self.id_num = int(id[-1]) + 1
            else:
                self.id_num = 1
                os.rename(f"{self.path_seleccionado_del_directorio}/{nombre_imagen}.png",
                          f"{self.path_seleccionado_del_directorio}/{nombre_imagen[0:5]}0{nombre_imagen[5:]}.png")

            self.forma_nombre_imagen()

    def inicializa_nombre_imagen(self):
        self.id_analisis = ""
        self.id_patogeno = ""
        self.id_tiempo = ""
        self.id_dilucion = ""
        self.id_num = ""
        self.id_zona = ""
        self.comboBox_zona.setCurrentIndex(0)
        self.comboBox_DOtro.setCurrentIndex(0)
        self.forma_nombre_imagen()

    def fn_boton_Reset(self):
        self.inicializa_nombre_imagen()

 # ============================================= AÑADIR / ELIMINAR FICHEROS ============================================
        """Mediante las siguientes funciones se añadir o eliminan ficheros y/o carpetas seleecionadas en el menu de archivos.
        Siempre se pedira confirmacion en caso de eliminar algo a través de un mensaje de alerta.
        """

    def fn_boton_anadir_carpeta(self):
        nombre_nueva_carpeta = self.edit_nombre_nueva_carpeta.text()
        if len(nombre_nueva_carpeta) != 0:
            index = self.lista_analisis.currentIndex()
            path_nueva_carpeta_aux = self.dirModel.filePath(index)
            path_nueva_carpeta = f"{path_nueva_carpeta_aux}/{nombre_nueva_carpeta}"

            try:
                os.mkdir(path_nueva_carpeta)

            except:
                texto = "La carpeta que intentas crear ya existe"
                crea_mensaje_alerta(texto)

        else:
            texto = "No has introducido ningun nombre de carpeta"
            crea_mensaje_alerta(texto)

    def fn_boton_borrar_fichero(self, event):

        borrar = QMessageBox.question(self,
                                      "ELIMINAR FICHERO",
                                      "¿ESTAS SEGURO DE ELIMINAR ESTE FICHERO?",
                                      QMessageBox.Yes | QMessageBox.No)
        if borrar == QMessageBox.Yes:
            if self.path_seleccionado_del_directorio[-4:] != ".png":
                shutil.rmtree(self.path_seleccionado_del_directorio)
                model_limpiar_lista_imagenes = QStandardItemModel()
                self.lista_imagenes.setModel(model_limpiar_lista_imagenes)
                model_limpiar_lista_imagenes.removeRow(0)

            else:
                os.remove(self.path_seleccionado_del_directorio)
        else:
            event.ignore()

 # ========================================= ACTIVAR DESACTIVAR CONJUNTO DE BOTONES ===================================
        """Aqui se configura la ativacion/desactivacion de los botones de la GUI dependiendo del modo de trabajo en el que
        estemos, ya sea evaluación, o captura de imagenes.
        """

    def activar_desactivar_botones_eleccion(self, FLAG_BOTONES):
        self.boton_AB.setEnabled(FLAG_BOTONES)
        self.boton_BC.setEnabled(FLAG_BOTONES)
        self.boton_CA.setEnabled(FLAG_BOTONES)
        self.boton_EC.setEnabled(FLAG_BOTONES)
        self.boton_PA.setEnabled(FLAG_BOTONES)
        self.boton_SA.setEnabled(FLAG_BOTONES)
        self.boton_MO.setEnabled(FLAG_BOTONES)
        self.boton_LE.setEnabled(FLAG_BOTONES)
        self.boton_AM.setEnabled(FLAG_BOTONES)
        self.boton_D1.setEnabled(FLAG_BOTONES)
        self.boton_D2.setEnabled(FLAG_BOTONES)
        self.boton_D3.setEnabled(FLAG_BOTONES)
        self.comboBox_DOtro.setEnabled(FLAG_BOTONES)
        self.boton_T0.setEnabled(FLAG_BOTONES)
        self.boton_T7.setEnabled(FLAG_BOTONES)
        self.boton_T14.setEnabled(FLAG_BOTONES)
        self.boton_T28.setEnabled(FLAG_BOTONES)
        self.comboBox_zona.setEnabled(FLAG_BOTONES)

    def activar_desactivar_botones_edicion(self, FLAG_BOTONES):
        self.boton_ver.setEnabled(FLAG_BOTONES)
        self.boton_ocultar.setEnabled(FLAG_BOTONES)
        self.boton_lapiz.setEnabled(FLAG_BOTONES)
        self.boton_borrar.setEnabled(FLAG_BOTONES)
        # self.boton_validar.setEnabled(FLAG_BOTONES)
        # self.boton_cancelar.setEnabled(FLAG_BOTONES)
        self.boton_ausencia.setEnabled(FLAG_BOTONES)
        self.boton_presencia.setEnabled(FLAG_BOTONES)
        self.boton_Evaluar.setEnabled(FLAG_BOTONES)

 # ================================================= VER / OCULTAR RESULTADO ==========================================
        """Estas funciones permiten ocultar/mostrar las regiones de una imagen en caso de haberlas.
        """

    def fn_boton_ver(self):
        self.coloca_imagen_seleccionada_en_visor(
            path=self.path_seleccionado_del_directorio_modificado)
        self.boton_ver.setDisabled(True)
        self.boton_ocultar.setEnabled(True)

    def fn_boton_ocultar(self):
        self.coloca_imagen_seleccionada_en_visor(
            path=self.path_seleccionado_del_directorio)
        self.boton_ocultar.setDisabled(True)
        self.boton_ver.setEnabled(True)

 # ================================================ FUNCIONES INFERENCIA ==============================================
        """En este apartado se ejecuta el pipeline asociado a la inferencia de la imagen seleccionada. Se ha de clicar
        en el boton evaluar, y entonces se ejecuta un multihilo que a su vez ejecuta distintas funciones. Entre ellas,
        elegir el modelo de prediccion, determinar si la tarea es de deteccion y clasificacion, y guardar el resultado
        en una nueva imagen/fichero. Además, se ha añadido una progressBar que se actualiza a medida que las imagenes
        se infernecias, ya que se puede inferenciar una unica placa si se selecciona la imagen, o todas las placas
        contenidas en la carpeta si se selecciona la carpeta.
        """

    def fn_boton_Evaluar(self):

        self.progresBar.setValue(0)
        Main.setDisabled(self, True)
        self.prediccionQThread = CalculaPrediccion(
            self.path_seleccionado_del_directorio)
        self.prediccionQThread.envio_prediccion.connect(self.recibePrediccion)
        self.prediccionQThread.envio_valor_progressBar.connect(
            self.maneja_valor_progresBar)
        self.prediccionQThread.start()

    def recibePrediccion(self, prediccion):
        "Se obtiene un diccionario con los paths de las imagenes y la prediccion"

        self.prediccionQThread.stop()
        try:
            guarda_ImagenFichero_prediccion(prediccion)
        except:
            texto = "Todas las placas han sido ya evaluadas"
            Main.setDisabled(self, False)
            self.progresBar.setValue(0)
            crea_mensaje_alerta(texto)

        Main.setDisabled(self, False)
        self.progresBar.setValue(100)

    def maneja_valor_progresBar(self, valor):
        self.progresBar.setValue(valor)

 # ============================================== FUNCIONES CERRAR VENTANA ============================================
        """A traves de estas funciones se define la logica de cierre de ventana.
        """

    def closeEvent(self, event):
        """
        Cerrar la ventana a través de el boton X de la esquina supeior derecha

        :param event: evento del raton
        :return: None
        """
        close = QMessageBox.question(self,
                                     "SALIR",
                                     "¿ESTAS SEGURO/A DE QUERER SALIR?",
                                     QMessageBox.Yes | QMessageBox.No)
        if close == QMessageBox.Yes:
            event.accept()
            exit()
        else:
            event.ignore()

    def keyPressEvent(self, e):
        """
        Cerrar la ventana a traves de la tecla ESC

        :param e: evento del teclado
        :return: None
        """
        if e.key() == QtCore.Qt.Key_Escape:
            self.closeEvent(e)
            self.close()
        if e.key() == QtCore.Qt.Key_F11:
            if self.isMaximized():
                self.showNormal()
            else:
                self.showMaximized()

 # =================================================== MODO EDICION ====================================================
        """Es en el actual apartado se configuran las funciones de las botoneras de edicion. Para ello, se tiene en cuenta
        la pulsacion del raton. Si se selecciona el lapiz, se pintan las regiones en la pantalla mediante dos clicks y 
        despues se han de confirmar los cambios. Esta confirmacion generara un nuevo archivo con las coordenadas de las regiones
        y una nueva imagen pintada. En caso de utilizar la herramienta de borrar, se selecciona la region a borrar y posteriormente
        se confirma, eliminando asi las respectivas coordenadas. Mediante los botones de ausencia y presencia, y se elige la 
        clase a la cual pertene la placa que se esta editando. 
        """

    def fn_boton_lapiz(self):
        self.activar_desactivar_botones_edicion(False)
        self.boton_validar.setDisabled(False)
        self.boton_cancelar.setDisabled(False)
        self.FLAG_DIBUJAR = True
        self.FLAG_VALIDAR_EDICION_DETECCION = True

    def fn_boton_borrar(self):
        self.activar_desactivar_botones_edicion(False)
        self.boton_validar.setDisabled(False)
        self.boton_cancelar.setDisabled(False)
        self.FLAG_BORRAR = True
        self.FLAG_VALIDAR_EDICION_DETECCION = True

    def fn_boton_ausencia(self):
        self.activar_desactivar_botones_edicion(False)
        self.boton_validar.setDisabled(False)
        self.boton_cancelar.setDisabled(False)
        self.nueva_clase = 'AUSENCIA'
        self.FLAG_VALIDAR_EDICION_CLASIFICACION = True

    def fn_boton_presencia(self):
        self.activar_desactivar_botones_edicion(False)
        self.boton_validar.setDisabled(False)
        self.boton_cancelar.setDisabled(False)
        self.nueva_clase = 'PRESENCIA'
        self.FLAG_VALIDAR_EDICION_CLASIFICACION = True

    def fn_boton_validar(self):
        self.activar_desactivar_botones_edicion(True)
        self.boton_validar.setDisabled(True)
        self.boton_cancelar.setDisabled(True)

        if self.FLAG_DIBUJAR:
            nuevo_resultado = self.lista_nuevas_boxes
        elif self.FLAG_BORRAR:
            nuevo_resultado = self.lista_borrar_boxes
        elif self.FLAG_VALIDAR_EDICION_CLASIFICACION:
            nuevo_resultado = self.nueva_clase

        valida_cambios_edicion(
            nuevo_resultado, path_archivo=self.path_seleccionado_del_directorio_modificado, flag_dibujar=self.FLAG_DIBUJAR, flag_borrar=self.FLAG_BORRAR, flag_edicion_clasificacion=self.FLAG_VALIDAR_EDICION_CLASIFICACION)

        self.coloca_imagen_seleccionada_en_visor(
            path=self.path_seleccionado_del_directorio_modificado)

        self.label_resultado_prediccion.setText(muestra_resultado_enLabel(
            self.path_seleccionado_del_directorio_modificado))

        self.lista_nuevas_boxes = []
        self.lista_borrar_boxes = []
        self.FLAG_DIBUJAR = False
        self.FLAG_BORRAR = False
        self.FLAG_VALIDAR_EDICION_CLASIFICACION = False

    def fn_boton_cancelar(self):
        self.activar_desactivar_botones_edicion(True)
        self.boton_validar.setDisabled(True)
        self.boton_cancelar.setDisabled(True)

        self.coloca_imagen_seleccionada_en_visor(
            path=self.path_seleccionado_del_directorio_modificado)
        self.label_resultado_prediccion.setText(muestra_resultado_enLabel(
            self.path_seleccionado_del_directorio_modificado))

        self.lista_nuevas_boxes = []
        self.lista_borrar_boxes = []
        self.FLAG_DIBUJAR = False
        self.FLAG_BORRAR = False
        self.FLAG_VALIDAR_EDICION_DETECCION = False
        self.FLAG_VALIDAR_EDICION_CLASIFICACION = False

    def mousePressEvent(self, event):

        if self.FLAG_DIBUJAR:

            if self.FLAG_PRIMER_CLICK:
                self.primer_click = self.visor.mapFromParent(event.pos())
                self.primer_click_x = self.primer_click.x()
                self.primer_click_y = self.primer_click.y()

                if comprueba_click_dentro_imagen(self.primer_click_x, self.primer_click_y, self.ancho_zonaClick_imagen, self.alto_zonaClick_imagen):

                    self.FLAG_PRIMER_CLICK = False
                    self.FLAG_SEGUNDO_CLICK = True

            elif self.FLAG_SEGUNDO_CLICK:
                self.segundo_click = self.visor.mapFromParent(event.pos())
                self.segundo_click_x = self.segundo_click.x()
                self.segundo_click_y = self.segundo_click.y()

                if comprueba_click_dentro_imagen(self.segundo_click_x, self.segundo_click_y, self.ancho_zonaClick_imagen, self.alto_zonaClick_imagen):

                    self.FLAG_PRIMER_CLICK = True
                    self.FLAG_SEGUNDO_CLICK = False

                    dibuja_region(QPainter(self.visor.pixmap()), self.primer_click_x, self.primer_click_y,
                                  self.segundo_click_x, self.segundo_click_y, 'green', 3, self.offset_X, self.offset_Y)

                    self.update()
                    self.visor.update()

                    # Guardamos en una lista la nueva region añadida
                    box_a_anadir = [int(min([self.primer_click_x-self.offset_X, self.segundo_click_x-self.offset_X]) / self.factor_escala_dimension_x),
                                    int(min([self.primer_click_y-self.offset_Y, self.segundo_click_y-self.offset_Y]
                                            ) / self.factor_escala_dimension_y),
                                    int(max([self.primer_click_x-self.offset_X, self.segundo_click_x-self.offset_X]
                                            ) / self.factor_escala_dimension_x),
                                    int(max([self.primer_click_y-self.offset_Y, self.segundo_click_y-self.offset_Y])/self.factor_escala_dimension_y)]

                    self.lista_nuevas_boxes.append(box_a_anadir)

        elif self.FLAG_BORRAR:
            self.click_borrar = self.visor.mapFromParent(event.pos())
            self.click_borrar_x = self.click_borrar.x()
            self.click_borrar_y = self.click_borrar.y()

            if comprueba_click_dentro_imagen(self.click_borrar_x, self.click_borrar_x, self.ancho_zonaClick_imagen, self.alto_zonaClick_imagen):

                coordenadas_click = [
                    int(self.click_borrar_x), int(self.click_borrar_y)]

                box_a_borrar, click_correcto = borrar_region(QPainter(self.visor.pixmap(
                )), coordenadas_click, self.path_seleccionado_del_directorio_modificado, 'red', 3, self.offset_X, self.offset_Y, self.factor_escala_dimension_x, self.factor_escala_dimension_y)

                if click_correcto:
                    self.update()
                    self.visor.update()
                    self.lista_borrar_boxes.append(box_a_borrar)

    # def paintEvent(self, event):
    #     if self.FLAG_DIBUJAR:
    #         qp = QPainter(self.visor.pixmap())
    #         br = QtGui.QBrush(QColor(100, 10, 10, 40))
    #         qp.setBrush(br)
    #         qp.drawRect(QtCore.QRect(self.begin, self.end))

    # def mousePressEvent(self, event):
    #     self.begin = self.visor.mapFromParent(event.pos())
    #     self.end = self.visor.mapFromParent(event.pos())
    #     self.update()
    #     # self.visor.update()

    # def mouseMoveEvent(self, event):
    #     self.end = self.visor.mapFromParent(event.pos())
    #     self.update()
    #     # self.visor.update()

    # def mouseReleaseEvent(self, event):
    #     self.begin = self.visor.mapFromParent(event.pos())
    #     self.end = self.visor.mapFromParent(event.pos())
    #     self.update()
    #     self.visor.update()

# =================================================== TABLA RESULTADOS ==================================================

    def fn_mostrarItem(self, fila, columna):
        try:
            print(self.tabla.item(fila, columna).text())
        except:
            pass

    def ejecuta_query(self, sqlStatement):
        self.qry.prepare(sqlStatement)
        self.qry.exec()

        self.fn_muestra_datos(self.qry)

    def fn_muestra_datos(self, query):

        self.modelo_query.setQuery(query)
        self.tabla.setModel(self.modelo_query)
        self.tabla.show()


# ================================================== INICIA APLICACION ================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    # app.setStyleSheet(qdarkstyle.load_stylesheet())
    window = Main()
    window.show()
    window.showMaximized()
    sys.exit(app.exec_())
