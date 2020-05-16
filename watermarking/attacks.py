"""This script is used to digitally transform watermarked image."""

from io import BytesIO
from cv2 import (
    medianBlur,
    blur,
    GaussianBlur,
    resize,
    getRotationMatrix2D,
    warpAffine # continuity of rotation
)
from skimage.util import random_noise as noise
from numpy import array # to get shape of image
from PIL import Image

class Attacks:
    """Contains attacks for watermarked image."""

    def __init__(self, image):
        self.image = image

    def median_blur(self, filter_size):
        """Perform median blur on image with particular filter size"""
        return medianBlur(self.image, filter_size)

    def average_filter(self, filter_shape):
        """Perform average filter on image with particular filter size"""
        return blur(self.image, filter_shape)

    def gaussian_blur(self, filter_shape):
        """Perform gaussian blur on image with particular filter size"""
        return GaussianBlur(self.image, filter_shape, 0)

    def resize(self, percentage):
        """Scale up/down image based on defined percentage"""
        return resize(
            self.image,
            self.get_size_from_percentage(percentage)
        )

    def get_size_from_percentage(self, percentage):
        """Get size of defined percentage"""
        image = array(self.image)
        width = int(image.shape[1] * percentage / 100)
        height = int(image.shape[0] * percentage / 100)
        return (width, height)

    def crop(self, percentage):
        """Crop image based on defined percentage"""
        size = self.get_size_from_percentage(percentage)
        return self.image[
            (0 + size[0]):(len(self.image) - size[0] - 1),
            (0 + size[1]):(len(self.image) - size[1] - 1)
        ]

    def rotation(self, degree):
        """Rotate image as much as defined degree"""
        image = array(self.image)
        rotation = getRotationMatrix2D(
            self.get_center_point(image),
            degree,
            1 # this is the scale
        )
        return warpAffine(image, rotation, image.shape[:2])

    def get_center_point(self, image):
        """Get center point of an image"""
        return (image.shape[:2][0] / 2, image.shape[:2][1] / 2)

    def gaussian_noise(self, variance):
        """Gave noise to image with defined variance"""
        return array(255 * noise(self.image, mode='gaussian', var=variance), dtype='uint8')

    def salt_and_pepper(self, amount):
        """Gave noise to image with defined amount"""
        return array(255 * noise(self.image, mode='s&p', amount=amount), dtype='uint8')

    # this one is not gonna using image from self because the image type should be PIL
    # the others use numpy array image
    def compress_jpeg(self, image, quality):
        """Return PIL JPEG-compressed image."""
        buffer = BytesIO()
        image.save(buffer, format="jpeg", quality=quality)
        buffer.seek(0)
        return Image.open(BytesIO(buffer.read()))
