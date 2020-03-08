import cv2
import numpy as np
from PIL import Image
from watermarking import attacks
from watermarking import embedding

class Process:
    ROOT = "static/"
    HOST = "host.tiff"
    WM = "wm.tiff"
    WMED = "wmed.tiff"
    PREVIEW_HOST = "data/preview_host.jpg"
    PREVIEW_WM = "data/preview_wm.jpg"
    PREVIEW_WMED = "data/preview_wmed.jpg"

    def __init__(self):
        self.host = None
        self.watermark = None
        self.watermarked = None

    # util functions

    # function to decode js image to open cv format
    def jsImageToOpenCV(self, byteImg):
        return cv2.imdecode(np.frombuffer(byteImg, dtype=np.uint8), cv2.IMREAD_UNCHANGED)

    def pilToOpenCV(self, img):
        pil_image = img.convert('RGB')
        open_cv_image = np.array(pil_image)
        return open_cv_image[:, :, ::-1].copy()

    def openCVToPIL(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img)

    #watermarking functions

    def training(self):
        print('tes')
    
    def embed(self):
        embedding.Embedding().embed(self.host, self.watermark)
        #everytime embedding is finished, clear host and watermark from system

    def extract(self):
        print('extract')

    #getter

    # function to save image and return its preview for html
    def getPreviewHost(self, img):
        self.host = self.jsImageToOpenCV(img)
        return self.createPreview(img, self.PREVIEW_HOST)

    def getPreviewWM(self, img):
        self.watermark = self.jsImageToOpenCV(img)
        return self.createPreview(img, self.PREVIEW_WM)

    def getPreviewWMED(self, img):
        self.watermarked = self.jsImageToOpenCV(img)
        return self.createPreview(img, self.PREVIEW_WMED)

    def createPreview(self, img, filename):
        transformation = attacks.Attacks()
        transformation.compressJPEG(
                self.openCVToPIL(self.jsImageToOpenCV(img)),
                90
        ).save(self.ROOT + filename)
        return filename