import os


class cv2upscaler:
    import cv2
    import os
    Labels = ["По соседям", 'Билинейная', 'Бикубическая', 'Inter_Area', 'Lanczos4',
              'EDSR', 'ESPCN', 'FSRCNN', 'LapSRN']

    interpolationMethod = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4]
    interpolationModels = ['EDSR_x2.pb', 'ESPCN_x2.pb', 'FSRCNN_x2.pb', 'LapSRN_x2.pb']
    modelNames = ['edsr', 'espcn', 'fsrcnn', 'lapsrn']
    sr = cv2.dnn_superres.DnnSuperResImpl_create()

    def ImageUpscaleClassic(self, method, imagepath, razmer):
        img = self.cv2.imread(imagepath)
        img = self.cv2.resize(img, (razmer * 2, razmer * 2), interpolation=self.interpolationMethod[method])
        self.cv2.imwrite(imagepath, img)

    def ImageUpscaleNeuro(self, model, modelname, imagepath):
        modeldir = os.path.join(os.getcwd(), 'Models', 'OpenCV2')
        self.sr.readModel(self.os.path.join(modeldir, str(model)))
        self.sr.setModel(modelname, 2)
        img = self.cv2.imread(imagepath)
        self.cv2.imwrite(imagepath, self.sr.upsample(img))

    def ChooseModel(self, name, imagepath, razmer):
        if name == "По соседям":
            self.ImageUpscaleClassic(self, method=self.interpolationMethod[0], imagepath=imagepath, razmer=razmer)
        elif name == "Билинейная":
            self.ImageUpscaleClassic(self,method=self.interpolationMethod[1], imagepath=imagepath, razmer=razmer)
        elif name == "Бикубическая":
            self.ImageUpscaleClassic(self,method=self.interpolationMethod[2], imagepath=imagepath, razmer=razmer)
        elif name == "Inter_Area":
            self.ImageUpscaleClassic(self,method=self.interpolationMethod[3], imagepath=imagepath, razmer=razmer)
        elif name == "Lanczos4":
            self.ImageUpscaleClassic(self,method=self.interpolationMethod[4], imagepath=imagepath, razmer=razmer)
        elif name == "EDSR":
            self.ImageUpscaleNeuro(self, model=self.interpolationModels[0], modelname=self.modelNames[0], imagepath=imagepath)
        elif name == "ESPCN":
            self.ImageUpscaleNeuro(self, model=self.interpolationModels[1], modelname=self.modelNames[1], imagepath=imagepath)
        elif name == "FSRCNN":
            self.ImageUpscaleNeuro(self, model=self.interpolationModels[2], modelname=self.modelNames[2], imagepath=imagepath)
        elif name == "LapSRN":
            self.ImageUpscaleNeuro(self, model=self.interpolationModels[3], modelname=self.modelNames[3], imagepath=imagepath)
