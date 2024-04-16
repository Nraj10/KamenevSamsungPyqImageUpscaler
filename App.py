from PyQt5 import QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QGraphicsPixmapItem, QGraphicsScene

import design
import os

class App(QtWidgets.QMainWindow, design.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.TileSize.setValue(100)
        self.InputImageButton.clicked.connect(self.getImagePath)

    def getImagePath(self):
        path = QtWidgets.QFileDialog.getOpenFileName(self, "Выберите изображение", filter='*.png, *.jpg')
        self.loadImageInBrowser(path)

    def loadImageInBrowser(self, path):
        image_qt = QImage(path[0])
        pic = QGraphicsPixmapItem()
        pic.setPixmap(QPixmap.fromImage(image_qt))
        scene = QGraphicsScene()
        scene.addItem(pic)
        self.ImageBrowser.setScene(scene)

