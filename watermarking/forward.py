"""This script is to process data with forward CNN"""

from numpy import add, flip, array
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
    softmax_kernels = {}

    # training cache
    convolution_cache = []
    batch_norm_cache = []
    relu_cache = []
    max_pooling_cache = []
    conv_softmax_cache = []
    softmax_cache = []

    def __init__(self, istraining, inputs, params):
        self.istraining = istraining
        self.inputs = array(inputs)
        print(self.inputs.shape)
        (
            self.scale_shift,
            self.encoder_kernels,
            self.decoder_kernels,
            self.softmax_kernels
        ) = params

    def run(self):
        """Function that will be run from other classes"""
        if self.istraining:
            self.init_cache()

        encoded = []
        # encoder
        print('encoder', flush=True)
        for batch, image in enumerate(self.inputs):
            print('shape: ', array(image).shape)
            for i in range(len(cnn.CNN.CONVOLUTION_ORDERS[cnn.CNN.ENCODER])):
                image = self.conv_per_stack(
                    image, self.ENCODER, i, batch
                )
                if self.istraining:
                    image = self.batch_norm_per_stack(
                        image,
                        cnn.CNN.ENCODER,
                        i
                    )
                image = self.relu_per_stack(image, batch)
                image = self.max_pooling_per_stack(image, batch)
            encoded.append(image)
        decoded = []
        #decoder
        print('decoder', flush=True)
        for batch, image in enumerate(encoded):
            for i in range(
                    len(cnn.CNN.CONVOLUTION_ORDERS[cnn.CNN.DECODER]) - 1,
                    -1, -1
                ):
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
                        i
                    )
            decoded.append(image)

        return self.softmax_per_batch(decoded), (
            self.convolution_cache,
            self.batch_norm_cache,
            self.relu_cache,
            self.max_pooling_cache,
            self.conv_softmax_cache,
            self.softmax_cache
        )

    def init_cache(self):
        """Initialize cache for each backprop later"""
        print('initializing')
        self.init_convolution_cache()
        self.init_two_dimension_cache()

    def init_convolution_cache(self):
        """Initialize convolutions cache.
           Structure can be seen in notes."""
        for _ in range(training.Training.BATCH_SIZE):
            member_arr = []
            for part in range(self.CNN_PART):
                part_arr = []
                str_part = cnn.CNN.ENCODER if part == 0 else cnn.CNN.ENCODER
                for stack in range(0, len(cnn.CNN.CONVOLUTION_ORDERS[str_part])):
                    stack_arr = []
                    for ____ in range(
                            0, cnn.CNN.CONVOLUTION_ORDERS[str_part][stack][0]
                        ): # layer
                        stack_arr.append([])
                    part_arr.append(stack_arr)
                member_arr.append(part_arr)
            self.convolution_cache.append(member_arr)

    def init_two_dimension_cache(self):
        """Initialize ReLU, batch norm, and max pooling cache"""
        for _ in range(training.Training.BATCH_SIZE):
            self.batch_norm_cache.append([])
            self.relu_cache.append([])
            self.max_pooling_cache.append([])

    # manually tested in jupyter
    def conv_per_stack(self, matrices, part, stack_number, batch_number):
        """Convolution as many as number of layers and channels in current stack number.
           Returning a bunch of feature maps."""
        kernels = {}
        str_part = ''
        if part == self.DECODER:
            kernels = self.decoder_kernels
            str_part = cnn.CNN.DECODER
        else:
            kernels = self.encoder_kernels
            str_part = cnn.CNN.ENCODER
        for layer in range(
                cnn.CNN.CONVOLUTION_ORDERS[str_part][stack_number][0]
            ): #each stack consists of 2-3 layers
            feature_maps = []
            if self.istraining: #initialize cache for layer
                self.convolution_cache[
                    batch_number
                ][
                    part
                ][
                    stack_number
                ][
                    layer
                ] = matrices
            feature_map = []
            for channel in range(
                    cnn.CNN.CONVOLUTION_ORDERS[str_part][stack_number][1]
                ): #each layer consists of 64-512 channels
                for matrix in matrices:
                    convolved = convolve2d(
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
                    feature_map = convolved if(
                        channel == 0
                    ) else add(feature_map, convolved)
                feature_maps.append(feature_map)
            matrices = feature_maps
        return matrices

    def batch_norm_per_stack(self, matrices, part, stack_number):
        """Process each batch member with Batch Normalization"""
        print('BN', flush=True)
        print(array(matrices).shape)
        normalized, cache = cnn.CNN.batch_norm(
            matrices,
            self.scale_shift[part + "-" + str(stack_number) + "-beta"],
            self.scale_shift[part+ "-" + str(stack_number) + '-gamma']
        )
        self.batch_norm_cache.append(cache)
        print('cache: ', array(cache[0]).shape, ' for stack ', len(self.batch_norm_cache) - 1)
        return normalized

    def relu_per_stack(self, matrices, batch_number):
        """Process each batch member with ReLU"""
        print('ReLU', flush=True)
        relued = []
        if self.istraining:
            self.relu_cache[batch_number].append(matrices)
        for matrix in matrices:
            relued.append(cnn.CNN.relu(matrix))
        return relued

    def max_pooling_per_stack(self, matrices, batch_number):
        """Process each batch member with max pooling"""
        print('Max pool', flush=True)
        max_pooled = []
        cache = []
        for matrix in matrices:
            result, max_index = cnn.CNN.max_pooling(matrix)
            max_pooled.append(result)
            cache.append([len(matrix), len(matrix[0]), max_index])
        if self.istraining:
            self.max_pooling_cache[batch_number].append(cache)
        else:
            self.max_pooling_index.append(cache)
        return max_pooled

    def upsample_per_stack(self, matrices, batch_number, stack_number):
        """Process each batch member with upsampling"""
        matrices_indices = self.max_pooling_cache[
            batch_number
        ][
            stack_number
        ] if self.istraining else self.max_pooling_index[
            stack_number
        ]
        print('Ups', flush=True)
        upsampled = []
        i = 0
        try:
            for i, matrix in enumerate(matrices):
                upsampled.append(cnn.CNN.upsampling(
                    matrix,
                    matrices_indices[i]
                ))
        except IndexError:
            print('error at ', i)
            print(self.max_pooling_cache[0][3])
        return upsampled

    def softmax_per_batch(self, matrices):
        """Process each batch or single matrix into softmax output(s)"""
        if self.istraining:
            softmax_per_batch = []
            for matrix in matrices:
                self.conv_softmax_cache.append(matrix[0])
                result, cache = cnn.CNN.trainable_softmax(
                    self.softmax_kernels,
                    matrix[0] # has to access 0 because there seems
                    # to be an extra channel dimension
                )
                softmax_per_batch.append(result)
                self.softmax_cache.append(cache)
            return softmax_per_batch
        else:
            return cnn.CNN.trainable_softmax(
                self.softmax_kernels,
                matrices[0]
            )
