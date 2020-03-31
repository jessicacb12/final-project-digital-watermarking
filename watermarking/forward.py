"""This script is to process data with forward CNN"""

from numpy import add, flip
from scipy.signal import convolve2d
from watermarking import training, cnn

class Forward:
    """Process data per batch. If it's testing,
       then batch size is 1"""

    ENCODER = 0
    DECODER = 1
    CNN_PART = 2 #encoder and decoder

    istraining = False
    inputs = []
    max_pooling_index = []

    # params just to be used, not to be trained
    encoder_kernels = {}
    decoder_kernels = {}
    scale_shift = {}

    # training cache
    convolution_cache = []
    batch_norm_cache = []
    relu_cache = []
    max_pooling_cache = []
    softmax_cache = []

    def __init__(self, istraining, inputs, params):
        self.istraining = istraining
        self.inputs = inputs
        (
            self.scale_shift,
            self.encoder_kernels,
            self.decoder_kernels
        ) = params

    def run(self):
        """Function that will be run from other classes"""
        if self.istraining:
            self.init_cache()

        encoded = []
        # encoder
        print('encoder', flush=True)
        for batch, image in enumerate(self.inputs):
            for i in range(len(cnn.CNN.CONVOLUTION_ORDERS)):
                image = self.conv_per_stack(
                    image, self.ENCODER, i, batch
                )
                if self.istraining:
                    image = self.batch_norm_per_stack(
                        image,
                        cnn.CNN.ENCODER,
                        i,
                        batch
                    )
                image = self.relu_per_stack(image, batch)
                image = self.max_pooling_per_stack(image, batch)
            encoded.append(image)

        decoded = []
        #decoder
        print('decoder', flush=True)
        for batch, image in enumerate(encoded):
            for i in range(len(cnn.CNN.CONVOLUTION_ORDERS) - 1, -1, -1):
                image = self.upsample_per_stack(
                    image, batch, i
                )
                image = self.conv_per_stack(
                    image, self.DECODER, i, batch
                )
                if self.istraining:
                    image = self.batch_norm_per_stack(
                        image,
                        cnn.CNN.DECODER,
                        i,
                        batch
                    )
            decoded.append(image)
        # return self.softmax_into_single_output(self.inputs), (
        #     self.convolution_cache,
        #     self.batch_norm_cache,
        #     self.relu_cache,
        #     self.max_pooling_cache,
        #     self.softmax_cache
        # )
        return self.softmax_per_batch(decoded)

    def init_cache(self):
        """Initialize cache for each backprop later"""
        self.init_convolution_cache()
        self.init_two_dimension_cache()

    def init_convolution_cache(self):
        """Initialize convolutions cache.
           Structure can be seen in notes."""
        for _ in range(training.Training.BATCH_SIZE):
            member_arr = []
            for __ in range(self.CNN_PART):
                part_arr = []
                for ___ in range(len(cnn.CNN.CONVOLUTION_ORDERS)):
                    part_arr.append([])
                member_arr.append(part_arr)
            self.convolution_cache.append(member_arr)

    def init_two_dimension_cache(self):
        """Initialize ReLU, batch norm, and max pooling cache"""
        for _ in range(training.Training.BATCH_SIZE):
            self.batch_norm_cache.append([])
            self.relu_cache.append([])
            self.max_pooling_cache.append([])

    def conv_per_stack(self, matrix, part, stack_number, batch_number):
        """Convolution as many as number of layers and channels in current stack number"""
        print('conv: ', len(matrix), flush=True)
        kernels = {}
        str_part = ''
        if part == self.DECODER:
            kernels = self.decoder_kernels
            str_part = cnn.CNN.DECODER
        else:
            kernels = self.encoder_kernels
            str_part = cnn.CNN.ENCODER

        for layer in range(
                cnn.CNN.CONVOLUTION_ORDERS[stack_number][0]
            ): #each stack consists of 2-3 layers
            combined_feature_maps = []

            if self.istraining: #initialize cache for layer
                self.convolution_cache[
                    batch_number
                ][
                    part
                ][
                    stack_number
                ].append(matrix)

            for channel in range(
                    cnn.CNN.CONVOLUTION_ORDERS[stack_number][1]
                ): #each layer consists of 64-512 channels
                feature_map = convolve2d(
                    matrix,
                    flip(
                        kernels[
                            str_part +
                            str(stack_number) +
                            "-" +
                            str(layer) +
                            "-" +
                            str(channel)
                        ]
                    ),
                    mode='same'
                )

                combined_feature_maps = feature_map if(
                    channel == 0
                ) else add(combined_feature_maps, feature_map)
            matrix = combined_feature_maps
        return matrix

    def batch_norm_per_stack(self, matrix, part, stack_number, batch_number):
        """Process each batch member with Batch Normalization"""
        print('BN', flush=True)
        result, cache = cnn.CNN.batch_norm(
            matrix,
            self.scale_shift[part + "-" + stack_number + "-beta"],
            self.scale_shift[part+ "-" + stack_number + '-gamma']
        )
        self.batch_norm_cache[batch_number].append(cache)
        return result

    def relu_per_stack(self, matrix, batch_number):
        """Process each batch member with ReLU"""
        print('ReLU', flush=True)
        if self.istraining:
            self.relu_cache[batch_number].append(matrix)
        return cnn.CNN.relu(matrix)

    def max_pooling_per_stack(self, matrix, batch_number):
        """Process each batch member with max pooling"""
        print('Max pool', flush=True)
        max_pooled, max_index = cnn.CNN.max_pooling(matrix)
        cache = [len(matrix), len(matrix[0]), max_index]
        if self.istraining:
            self.max_pooling_cache[batch_number].append(cache)
        else:
            self.max_pooling_index.append(cache)
        return max_pooled

    def upsample_per_stack(self, matrix, batch_number, stack_number):
        """Process each batch member with upsampling"""
        indices = self.max_pooling_cache[
            batch_number
        ][
            stack_number
        ] if self.istraining else self.max_pooling_index[
            stack_number
        ]
        print('Ups', flush=True)
        return cnn.CNN.upsampling(
            matrix,
            indices
        )

    def softmax_per_batch(self, matrices):
        """Process each batch into averaged softmax output"""
        print('Softmax', flush=True)
        if self.istraining:
            softmax_per_batch = []
            for matrix in matrices:
                self.softmax_cache.append(matrix)
                softmax_per_batch.append(cnn.CNN.softmax(matrix))
            return softmax_per_batch
        else:
            return cnn.CNN.softmax(matrices)
