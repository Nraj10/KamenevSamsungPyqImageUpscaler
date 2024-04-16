from PyQt5 import QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QGraphicsPixmapItem, QGraphicsScene

import design
import os
import PythonScripts.Operations.OpenCv2ImageUpscaler
import PythonScripts.Operations.ImageSegmentationOperation
import PythonScripts.Operations.PytorchImageUpscaler


class App(QtWidgets.QMainWindow, design.Ui_MainWindow):
    imgPath = ""
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.TileSize.setValue(100)
        self.InputImageButton.clicked.connect(self.getImagePath)

    def getImagePath(self):
        imgPath = QtWidgets.QFileDialog.getOpenFileName(self, "Выберите изображение", filter='*.png, *.jpg')
        self.loadImageInBrowser(imgPath)

    def loadImageInBrowser(self, path):
        image_qt = QImage(path[0])
        pic = QGraphicsPixmapItem()
        pic.setPixmap(QPixmap.fromImage(image_qt))
        scene = QGraphicsScene()
        scene.addItem(pic)
        self.ImageBrowser.setScene(scene)


    def cv2upscalePipeline(self):
        if self.TileCheckBox.isChecked():
            return 0
        else:
            return 0


    def pytorchUpscalePipeLine(self):
        return 0