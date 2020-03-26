"""This script is to training data with CNN"""

from watermarking import cnn

class Training:
    """Contains attributes too"""
    BATCH_SIZE = 4
    training_batch = [] #input

    # trainable params
    encoder_kernels = {}
    decoder_kernels = {}
    scale_shift = {}

    def __init__(self, embedding_maps):
        self.divide_training_images([embedding_maps])

        # initializing params
        (
            self.scale_shift,
            self.encoder_kernels,
            self.decoder_kernels
        ) = cnn.CNN.init_params()

    def divide_training_images(self, images):
        """Divide training images into batches of BATCH_SIZE images"""
        if images is not None:
            i = 0
            while i < len(images):
                self.training_batch.append(
                    images[i : i + self.BATCH_SIZE]
                )
                i += self.BATCH_SIZE
