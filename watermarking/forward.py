"""This script is to process data with forward CNN"""

from numpy import add, flip, array, expand_dims, zeros
from scipy.signal import convolve2d
from watermarking import training, cnn
import tensorflow as tf

class Forward:
    """Process data per batch. If it's testing,
       then batch size is 1"""

    ENCODER = 0
    DECODER = 1
    CNN_PART = 2 #encoder and decoder

    istraining = False
    inputs = []

    # params just to be used, not to be trained
    encoder_kernels = {}
    decoder_kernels = {}
    scale_shift = {}
    softmax_kernels = {}

    current_kernel = None
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

        if self.istraining:
            self.convolution_cache = []
            self.batch_norm_cache = []
            self.relu_cache = []
            self.max_pooling_cache = []
            self.conv_softmax_cache = []
            self.softmax_cache = []
            self.init_cache()
        else:
            self.max_pooling_index = []

    def run(self):
        """Function that will be run from other classes"""
        encoded = []
        # encoder
        print('encoder', flush=True)
        print('shape: ', self.inputs.shape)
        encoded = self.inputs
        for i in range(len(cnn.CNN.CONVOLUTION_ORDERS[cnn.CNN.ENCODER])):
            print('STACK:', i)
            conved_images = []
            for batch, image in enumerate(encoded):
                conved_images.append(
                    self.conv_per_stack(
                        image, self.ENCODER, i, batch
                    )
                )
            conved_images = self.batch_norm_per_stack(
                conved_images,
                cnn.CNN.ENCODER,
                i
            )
            processed_conv = []
            print('ReLU and max pool')
            for batch, image in enumerate(conved_images):
                # image = self.relu_per_stack(image, batch)
                image = self.max_pooling_per_stack(image, batch)
                processed_conv.append(image)
            encoded = processed_conv
        decoded = encoded
        #decoder
        print('decoder', flush=True)
        print('shape: ', array(decoded).shape)
        for i in range(
                len(cnn.CNN.CONVOLUTION_ORDERS[cnn.CNN.DECODER]) - 1,
                -1, -1
            ):
            print('STACK:', i)
            processed = []
            print('ups and conv')
            for batch, image in enumerate(decoded):
                image = self.upsample_per_stack(
                    array(image), batch, i
                )
                print('ups: ', array(image).shape)
                image = self.conv_per_stack(
                    image, self.DECODER, i, batch
                )
                processed.append(image)
            decoded = self.batch_norm_per_stack(
                processed,
                cnn.CNN.DECODER,
                i
            )
            # decoded = array(processed)
            print(decoded.shape)
        if self.istraining:
            return self.softmax_per_batch(decoded), (
                self.convolution_cache,
                self.batch_norm_cache,
                self.relu_cache,
                self.max_pooling_cache,
                self.conv_softmax_cache,
                self.softmax_cache
            )
        else:
            return self.softmax_per_batch(decoded)

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
            self.relu_cache.append([])
            self.max_pooling_cache.append([])

    # manually tested in jupyter
    def conv_per_stack(self, matrices, part, stack_number, batch_number):
        """Convolution as many as number of layers and channels in current stack number.
           Returning a bunch of feature maps."""
        matrices = array([self.fix_reverse_shape_3d(matrices)])
        print('conv per stack: ', matrices.shape)
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
            Forward.current_kernel = self.fix_reverse_shape_4d(
                self.take_from_dict(
                    kernels,
                    (
                        str_part,
                        stack_number,
                        layer
                    )
                )
            )
            matrices = tf.keras.layers.Conv2D(
                cnn.CNN.CONVOLUTION_ORDERS[str_part][stack_number][1],
                cnn.CNN.CONVOLUTION_KERNEL_SIZE,
                input_shape=matrices.shape[1:],
                activation='relu',
                padding = 'same',
                kernel_initializer=Forward.get_current_kernel,
                use_bias=False
            )(matrices).numpy()
            print(matrices.shape)
        return Forward.reverse_shape(matrices[0])

    def take_from_dict(self, kernels, address):
        param = []
        str_part, stack_number, layer = address
        for channel in range(
            cnn.CNN.CONVOLUTION_ORDERS[str_part][stack_number][1]
        ):
            new_channel = []
            for i in range(
                cnn.CNN.CONVOLUTION_ORDERS[str_part][stack_number][2][layer]
            ):
                new_channel.append(kernels[
                    str_part +
                    str(stack_number) +
                    "-" +
                    str(layer) +
                    "-" +
                    str(channel) +
                    "-" +
                    str(i)
                ])
            param.append(new_channel)
        return array(param)

    @staticmethod
    def get_current_kernel(shape, dtype=None):
        return Forward.current_kernel

    @staticmethod    
    def reverse_shape(feature_map):
        """64, 64, 8 -> 8, 64, 64"""
        ch_number = feature_map.shape[2]
        new_fm = []
        for i in range(ch_number):
            new_channel = []
            for row in feature_map:
                new_row = []
                for _px in row:
                    new_row.append(_px[i])
                new_channel.append(new_row)
            new_fm.append(new_channel)
        return array(new_fm)

    def fix_reverse_shape_3d(self, feature_map):
        """877 -> 778"""
        (i_number, w, h) = feature_map.shape
        new_feature_map = zeros((w, h, i_number))
        for i in range(i_number):
            for j in range(w):
                for k in range(h):
                    new_feature_map[j, k, i] = feature_map[i, j, k]
        return array(new_feature_map)

    def fix_reverse_shape_4d(self, weight):
        """8177 -> 7718"""
        (i_number, c, w, h) = weight.shape
        new_weight = zeros((w, h, c, i_number))
        for i in range(c):
            for j in range(i_number):
                for k in range(w):
                    for l in range(h):
                        new_weight[k, l, i, j] = weight[j, i, k, l]
        return array(new_weight)

    def batch_norm_per_stack(self, matrices, part, stack_number):
        """Process each batch member with Batch Normalization"""
        matrices = array(matrices)
        normalized, cache = cnn.CNN.batch_norm(
            matrices,
            self.scale_shift[part + str(stack_number) + "-beta"],
            self.scale_shift[part + str(stack_number) + '-gamma'],
            average=self.scale_shift[part + str(stack_number) + "-average"],
            variance=self.scale_shift[part + str(stack_number) + "-variance"]
        )
        if self.istraining:
            self.batch_norm_cache.append(cache)
            print('cache: ', array(cache[0]).shape, ' for stack ', len(self.batch_norm_cache) - 1)
        return normalized

    def relu_per_stack(self, matrices, batch_number):
        """Process each batch member with ReLU"""
        relued = []
        if self.istraining:
            self.relu_cache[batch_number].append(matrices)
        return cnn.CNN.relu(matrices)

    def max_pooling_per_stack(self, matrices, batch_number):
        """Process each batch member with max pooling"""
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
        upsampled = tf.keras.layers.UpSampling2D()(
            array([self.fix_reverse_shape_3d(matrices)])
        ).numpy()
        return Forward.reverse_shape(upsampled[0])

    def softmax_per_batch(self, matrices):
        """Process each batch or single matrix into softmax output(s)"""
        matrices = array(matrices, dtype=float)
        if self.istraining:
            softmax_per_batch = []
            for matrix in matrices: # per batch member
                self.conv_softmax_cache.append(matrix)
                result, cache = cnn.CNN.trainable_softmax(
                    self.softmax_kernels,
                    matrix
                )
                softmax_per_batch.append(result)
                self.softmax_cache.append(cache)
            print('shape: ', array(self.softmax_cache).shape)
            return softmax_per_batch
        else:
            (foreground, background), _ = cnn.CNN.trainable_softmax(
                self.fix_reverse_shape_4d(
                    self.softmax_kernels
                ),
                array([self.fix_reverse_shape_3d(matrices[0])]) 
                # 0 because batch dimension will be there
            )
            result = cnn.CNN.softmax_classifier(
                foreground, background
            )
            # return result, foreground, background
            return result
