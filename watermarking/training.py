"""This script is to training data with CNN"""

from copy import deepcopy
from re import sub, search
from os import listdir, remove # to get list of training images and remove the read one
import numpy
from numpy import array, uint8, expand_dims
from PIL import Image
from watermarking import (
    process,
    embedding,
    extraction,
    attacks,
    cnn,
    forward,
    backward
)

class Training:
    """Divide training image before processing with forward,
       cross entropy loss, and backward"""
    BATCH_SIZE = 2
    training_batch = [] # input
    ground_truth_images = [] # to compare with

    PRE_TRAINING_PATH = 'static/pre_training'
    TRAINING_PATH = 'static/training'
    DEFAULT_WATERMARK = 'static/Watermark.tiff'

    TRANSFORMATIONS = {
        attacks.Attacks.MEDIAN_BLUR: [3, 5, 7],
        attacks.Attacks.AVERAGE_FILTER: [3, 5, 7],
        attacks.Attacks.GAUSSIAN_BLUR: [3, 5, 7],
        attacks.Attacks.RESIZE: [200, 50, 75],
        attacks.Attacks.CROP: [1, 4],
        attacks.Attacks.ROTATE: [90, 180, 270],
        attacks.Attacks.GAUSSIAN_NOISE: [0.001, 0.005, 0.01],
        attacks.Attacks.SALT_PEPPER: [0.01, 0.05, 0.1],
        attacks.Attacks.JPEG: [30, 40, 50, 70],
    }

    # trainable params
    params = None

    def __init__(self):
        # initializing params
        self.params = cnn.CNN.init_params()

    def auto_training(self):
        """Auto training from training directory"""
        training_images_filename = listdir(self.PRE_TRAINING_PATH)
        watermark = Image.open(self.DEFAULT_WATERMARK)
        attacked_watermarks = self.normalize_watermark(
            self.apply_transformations(
                watermark, iswatermark=True
            )
        )
        print('watermark: ', array(attacked_watermarks).shape)
        result, loss = None, None

        for filename in training_images_filename:
            image = Image.open(self.PRE_TRAINING_PATH + "/" + filename)
            full_path = self.TRAINING_PATH + "/" + sub(
                search(".tif+$", filename).group(), "", filename
            )
            embedding.Embedding().embed_watermark(
                process.Process.pil_to_open_cv(
                    image
                ),
                array(watermark, dtype=uint8),
                full_path
            )
            image = Image.open(full_path + ".tif")
            self.divide_training_images(
                self.get_embedding_maps(
                    self.apply_transformations(image),
                    extraction.Extraction().extract_key_from_pil_image(image)
                ),
                attacked_watermarks
            )
            result, loss = self.run()
        return result, loss

    def normalize_watermark(self, images):
        """Because the size of the watermark varies and there's PIL
           Image too, it is needed to be normalized"""
        normalized = []
        for image in images:
            image = array(image, dtype=uint8)
            image.resize((cnn.CNN.INPUT_SIZE, cnn.CNN.INPUT_SIZE))
            normalized.append(image)
        return normalized

    def apply_transformations(self, image, iswatermark=False):
        """Apply transformations to image"""
        attacked_image = [image] # the first is not the attacked one
        transformation = attacks.Attacks(
            process.Process.pil_to_open_cv(image) if (
                not iswatermark
            ) else array(image, dtype=uint8) # because if pil_to_open_cv
            # is run to the WM image, the image structure goes all wrong
        )
        for attack, attributes in self.TRANSFORMATIONS.items():
            for attr in attributes:
                transformed = transformation.do_transformation(
                    attack,
                    attr,
                    image
                )
                attacked_image.append(
                    transformed
                )
        return attacked_image

    def get_embedding_maps(self, images, key):
        """Turn images into array of embedding maps"""
        embedding_maps = []
        print('total: ', len(images))
        for image in images:
            extractor = extraction.Extraction()
            if not isinstance(image, numpy.ndarray):
                image = process.Process.pil_to_open_cv(image)
            embedding_maps.append(
                extractor.get_embedding_map(
                    image,
                    key
                )
            )
        return embedding_maps

    def divide_training_images(self, images, ground_truth):
        """Divide training images into batches of BATCH_SIZE images"""
        if images and ground_truth is not None:
            # to add channel dimension
            images = expand_dims(images, axis=1)
            ground_truth = expand_dims(ground_truth, axis=1)

            i = 0
            while i < len(images):
                batch = images[i : i + self.BATCH_SIZE]
                self.training_batch.append(
                    batch
                )
                self.ground_truth_images.append(
                    ground_truth[i : i + self.BATCH_SIZE]
                )
                i += self.BATCH_SIZE
            print('training: ', array(self.training_batch).shape)
            print('ground truth: ', array(self.ground_truth_images).shape)

    def cross_entropy_per_batch(
            self,
            images_per_batch,
            ground_truth_per_batch
        ):
        """Return losses per batch"""
        loss = []
        for i, image in enumerate(images_per_batch):
            loss.append(cnn.CNN.cross_entropy_loss(
                image[0],
                image[1],
                ground_truth_per_batch[i]
            ))
        return loss

    def run(self):
        """Method to be called for training"""
        result = []
        cache = None
        for i, batch in enumerate(self.training_batch):
            print("=============================")
            print("FORWARD FOR BATCH: ", i)
            print("=============================")
            result, cache = forward.Forward(
                True,
                deepcopy(batch),
                self.params
            ).run()
            loss = self.cross_entropy_per_batch(
                result,
                self.ground_truth_images[i]
            )
            print("------------------------------")
            print('LOSS: ', loss)
            print("------------------------------")
            print("BACKWARD")
            backward.Backward(
                result,
                cache,
                self.params,
                self.ground_truth_images[i]
            ).run()
        # cnn.CNN.store_param(self.params[0], "batch norm") # scale shift
        # cnn.CNN.store_param(self.params[1], "kernel") # encoder
        # cnn.CNN.store_param(self.params[2], "kernel") # decoder
        return result, loss
        # return result, loss, cache
