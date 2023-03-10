"""This script is used either to embed and extract watermark."""

import cv2
import numpy as np
from PIL import Image
import calendar, time # to get timestamp
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

    @staticmethod
    def pil_to_open_cv(img):
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

    def embed(self):
        """Function to embed watermark."""
        if self.host is None or self.watermark is None:
            return {
                "error": """Please upload your watermark
                and your image that will be embedded with watermark."""
            }
        timestamp = self.get_timestamp()
        embedded = embedding.Embedding().embed_watermark(
            self.host,
            self.watermark,
            self.ROOT + embedding.Embedding.FILENAME + timestamp
        )
        embedded["image"] = self.create_preview(
            embedded["image"],
            embedding.Embedding.FILENAME + timestamp
        )
        self.reset_input_data()
        return embedded

    def extract(self):
        """Function to extract watermark if watermarked image is uploaded and key exists."""
        if self.watermarked is None:
            return {
                "error": "Please upload your watermarked image"
            }

        if self.extracted_key is None:
            return {
                "error": "This image has no watermark or watermark key is lost"
            }

        result = extraction.Extraction().extract_watermark(
            self.watermarked,
            self.extracted_key
        )

        if isinstance(result, str):
            return {
                "error": result
            }
        return {
            "image": self.create_preview(
                np.array(result, dtype=np.uint8),
                extraction.Extraction.FILENAME
            )
        }

    def get_preview_host(self, img):
        """Function to save host image and return its preview for html."""
        self.host = self.js_image_to_open_cv(img)
        return self.create_preview(
            self.js_image_to_open_cv(img),
            self.PREVIEW_HOST + self.get_timestamp()
        )

    def get_preview_watermark(self, img):
        """Function to save watermark image and return its preview for html."""
        self.watermark = self.js_image_to_open_cv(img)
        return self.create_preview(
            self.js_image_to_open_cv(img),
            self.PREVIEW_WM + self.get_timestamp()
        )

    def get_preview_watermarked(self, img):
        """Function to save watermarked image, extract its key and """
        """return image preview for html."""
        self.extracted_key = extraction.Extraction.extract_key_from_image_description(
            img.decode(encoding='latin-1')
        )
        self.watermarked = self.js_image_to_open_cv(img)
        return self.create_preview(
            self.js_image_to_open_cv(img),
            self.PREVIEW_WMED + self.get_timestamp()
        )

    def create_preview(self, img, filename):
        """Function to create JPEG preview for html."""
        transformation = attacks.Attacks()
        transformation.do_transformation(
            transformation.JPEG,
            90,
            self.open_cv_to_pil(img)
        ).save(self.ROOT + filename + ".jpg")
        return filename

    def get_timestamp(self):
        """Call this function to prevent image caching in web"""
        return str(calendar.timegm(time.gmtime()))