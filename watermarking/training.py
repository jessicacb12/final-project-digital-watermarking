"""This script is to training data with CNN"""

from copy import deepcopy
from watermarking import cnn, forward, backward

class Training:
    """Divide training image before processing with forward,
       cross entropy loss, and backward"""
    BATCH_SIZE = 1
    training_batch = [] # input
    ground_truth_images = [] # to compare with

    # trainable params
    params = None

    def __init__(self, embedding_maps, ground_truth_images):
        self.divide_training_images(
            embedding_maps,
            ground_truth_images
        )

        # initializing params
        self.params = cnn.CNN.init_params()

    def divide_training_images(self, images, ground_truth):
        """Divide training images into batches of BATCH_SIZE images"""
        if images and ground_truth is not None:
            i = 0
            while i < len(images):
                [batch] = images[i : i + self.BATCH_SIZE]
                self.training_batch.append(
                    batch
                )
                self.ground_truth_images.append(
                    ground_truth[i : i + self.BATCH_SIZE]
                )
                i += self.BATCH_SIZE

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
            print("FORWARD")
            result, cache = forward.Forward(
                True,
                deepcopy(batch),
                self.params
            ).run()
            loss = self.cross_entropy_per_batch(
                result,
                self.ground_truth_images[i]
            )
            # print("BACKWARD")
            # self.params = backward.Backward(
            #     softmax_outputs,
            #     cache,
            #     self.params,
            #     self.ground_truth_images[i]
            # ).run()
        # cnn.CNN.store_param(self.params[0], "batch norm") # scale shift
        # cnn.CNN.store_param(self.params[1], "kernel") # encoder
        # cnn.CNN.store_param(self.params[2], "kernel") # decoder
        # return result, loss
        return result, loss, cache
