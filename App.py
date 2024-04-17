from PyQt5 import QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QGraphicsPixmapItem, QGraphicsScene

import design
import os
from PythonScripts.Operations.OpenCv2ImageUpscaler import cv2upscaler
from PythonScripts.Operations.ImageSegmentationOperation import tileOperations
from PythonScripts.Operations.PytorchImageUpscaler import pytorchUpscaler


class App(QtWidgets.QMainWindow, design.Ui_MainWindow):
    imgPath = ""
    outputDir = ""
    ext = ""
    tilenumber = 0

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.TileSize.setValue(400)
        self.InputImageButton.clicked.connect(self.getImagePath)
        self.StartInferenceButton.clicked.connect(self.Accept)
        for i in cv2upscaler.Labels:
            self.ModelComboBox.addItem(i, i)
        for i in pytorchUpscaler.Labels:
            self.ModelComboBox.addItem(i, i)

    def getImagePath(self):
        self.imgPath = QtWidgets.QFileDialog.getOpenFileName(self, "Выберите изображение", filter='*.png *.jpg')[0]
        self.outputDir = os.path.dirname(self.imgPath) + "/UpscaleOutput"
        trash, self.ext = os.path.splitext(self.imgPath)

        self.loadImageInBrowser(self.imgPath, self.ImageBrowser)

    def loadImageInBrowser(self, path, browser):
        image_qt = QImage(path)
        pic = QGraphicsPixmapItem()
        pic.setPixmap(QPixmap.fromImage(image_qt))
        scene = QGraphicsScene()
        scene.addItem(pic)
        browser.setScene(scene)

    def Accept(self):
        if self.ModelComboBox.currentText() in cv2upscaler.Labels:
            self.cv2upscalePipeline()
        else:
            self.pytorchUpscalePipeLine()
        self.loadImageInBrowser(self.outputDir+"/Output.png", self.ImageBrowserOut)
        self.textBrowser.append("Выходной файл - " + self.outputDir + "Output.png")
        self.textBrowser.append("Количество тайлов - " +
                                str(self.tilenumber))



    def cv2upscalePipeline(self):
        outputDir = self.outputDir
        tilenum = tileOperations.tile(self=tileOperations, dir_in=outputDir, filename=self.imgPath, imageWidght=self.TileSize.value())
        self.tilenumber = tilenum
        filepaths = self.GetImagePaths()
        for i in filepaths:
            cv2upscaler.ChooseModel(self=cv2upscaler, name=self.ModelComboBox.currentText(),
                                      imagepath=i, razmer=self.TileSize.value())

        tileOperations.merge(self=tileOperations, imageDirectory=outputDir + "/Output", tiles=tilenum)

    def GetImagePaths(self):
        imageList = os.listdir(self.outputDir + "/Output")
        labels = []
        for i in imageList:
            labels.append(os.path.splitext(i)[0])
        labels = [int(x) for x in labels]
        labels.sort()
        filepath = []
        for i in labels:
            filepath.append(self.outputDir + "/Output" '\\' + str(i) + self.ext)
        return filepath

    def pytorchUpscalePipeLine(self):
        outputDir = self.outputDir
        tilenum = tileOperations.tile(self=tileOperations, dir_in=outputDir, filename=self.imgPath, imageWidght=self.TileSize.value())
        self.tilenumber = tilenum
        filepaths = self.GetImagePaths()
        pytorchUpscaler.UpcsalerInference(self=pytorchUpscaler, imgpaths= filepaths, size=self.TileSize.value() )
        tileOperations.merge(self=tileOperations, imageDirectory=outputDir + "/Output", tiles=tilenum)
