class tileOperations:
    import math
    import os
    import PIL
    from PIL import Image
    from itertools import product
    # from alive_progress import alive_bar

    PIL.Image.MAX_IMAGE_PIXELS = 933120000

    def tile(self, dir_in, imageWidght):
        import shutil

        name, ext = self.os.path.splitext('input.png')
        img = self.Image.open(self.os.path.join(dir_in, 'input.png'))
        newDir = self.os.path.join(dir_in, 'Output')

        # Check if the directory already exists
        if not self.os.path.exists(newDir):
            self.os.makedirs(newDir)
        else:
            shutil.rmtree(newDir)
            self.os.makedirs(newDir)

        w, h = img.size

        grid = self.product(range(0, h - h % imageWidght, imageWidght), range(0, w - w % imageWidght, imageWidght))
        c = 0
        for i, j in grid:
            box = (j, i, j + imageWidght, i + imageWidght)
            out = self.os.path.join(newDir, f'{c}{ext}')
            img.crop(box).save(out)
            c = c + 1
        return self.math.floor(w / imageWidght)

    def merge(self, imageDirectory, tiles):
        vipsbin = r'C:\Users\Nraj\Desktop\vips-dev-8.15\bin'
        add_dll_dir = getattr(self.os, 'add_dll_directory', None)
        if callable(add_dll_dir):
            add_dll_dir(vipsbin)
        else:
            self.os.environ['PATH'] = self.os.pathsep.join((vipsbin, self.os.environ['PATH']))

        import pyvips

        imagList = self.os.listdir(imageDirectory)
        labels = []
        for i in imagList:
            labels.append(self.os.path.splitext(i)[0])
        labels = [int(x) for x in labels]
        labels.sort()
        filepath = []
        for i in labels:
            filepath.append(imageDirectory + '\\' + str(i) + '.png')

        images = [pyvips.Image.new_from_file(filename) for filename in filepath]
        final = pyvips.Image.arrayjoin(images, across=tiles)
        final.pngsave(imageDirectory + ".png")
