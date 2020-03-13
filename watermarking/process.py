"""This script is used either to embed and extract watermark."""

import cv2
import numpy as np
from PIL import Image
from watermarking import attacks
from watermarking import embedding
from watermarking import extraction

class Process:
    """This process will be called in app.py."""

    ROOT = "static/"
    HOST = "host.tiff"
    WM = "wm.tiff"
    WMED = "wmed.tiff"
    PREVIEW_HOST = "data/preview_host"
    PREVIEW_WM = "data/preview_wm"
    PREVIEW_WMED = "data/preview_wmed"

    def __init__(self):
        self.reset_input_data()

    def js_image_to_open_cv(self, byte_img):
        """Function to decode js image to open cv format."""
        return cv2.imdecode(np.frombuffer(byte_img, dtype=np.uint8), cv2.IMREAD_UNCHANGED)

    def pil_to_open_cv(self, img):
        """Function to convert PIL to numpy array image."""
        pil_image = img.convert('RGB')
        open_cv_image = np.array(pil_image)
        return open_cv_image[:, :, ::-1].copy()

    def open_cv_to_pil(self, img):
        """Function to convert numpy array to PIL image."""
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return Image.fromarray(img)

    def reset_input_data(self):
        """Method to reset data so that new request = new processing"""
        self.host = None
        self.watermark = None
        self.watermarked = None
        self.extracted_key = None

    def training(self):
        """Function to training watermark extractor."""
        print('tes')

    def embed(self):
        """Function to embed watermark."""
        if self.host is None or self.watermark is None:
            return {}

        embedded = embedding.Embedding().embed_watermark(
            self.host,
            self.watermark
        )
        embedded["image"] = self.create_preview(
            embedded["image"],
            embedding.Embedding.FILENAME
        )
        self.reset_input_data()
        return embedded

    def extract(self):
        """Function to extract watermark."""
        extraction.Extraction().extract_watermark(self.watermarked)

    def get_preview_host(self, img):
        """Function to save host image and return its preview for html."""
        self.host = self.js_image_to_open_cv(img)
        return self.create_preview(
            self.js_image_to_open_cv(img),
            self.PREVIEW_HOST
        )

    def get_preview_watermark(self, img):
        """Function to save watermark image and return its preview for html."""
        self.watermark = self.js_image_to_open_cv(img)
        return self.create_preview(
            self.js_image_to_open_cv(img),
            self.PREVIEW_WM
        )

    def get_preview_watermarked(self, img):
        """Function to save watermarked image, extract its key and """
        """return image preview for html."""
        self.extracted_key = extraction.Extraction.extract_key_from_image_description(
            img.decode(encoding='latin-1')
        )

        self.watermarked = self.js_image_to_open_cv(img)
        print(self.watermarked.shape)
        return self.create_preview(
            self.js_image_to_open_cv(img),
            self.PREVIEW_WMED
        )

    def create_preview(self, img, filename):
        """Function to create JPEG preview for html."""
        transformation = attacks.Attacks()
        transformation.compress_jpeg(
            self.open_cv_to_pil(img),
            90
        ).save(self.ROOT + filename + ".jpg")
        return filename
