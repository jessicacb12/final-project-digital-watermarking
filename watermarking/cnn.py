"""This script is to process data with CNN either training or testing"""

from numpy import (
    random,
    sqrt,
    mean,
    var,
    exp,
    array,
    multiply,
    log,
    float32,
    zeros,
    ones,
    flip,
    equal
)
# from numpy import max as max_from_array
from numpy import sum as sum_array
from scipy.signal import convolve2d
from watermarking import value_with_position, forward
import tensorflow as tf

class CNN:
    """Keeps atrribute embedding map and weight"""

    INPUT_SIZE = 64
    CONVOLUTION_KERNEL_SIZE = 7
    POOLING_KERNEL_SIZE = 2
    POOLING_STRIDE = 2
    PADDING_SIZE = 3
    CONVOLUTION_ORDERS = {
        "enc": [
            [2, 8, [1, 8]],
            #[2, 8],
            # [3, 256],
            # [3, 512],
            # [3, 512]
        ],
        "dec": [
            [2, 8, [8, 8]],
            #[2, 4],
            # [3, 128],
            # [3, 256],
            # [3, 512]
        ],
        "softmax": [
            ['fg', 'bg'], 8
        ]
    }
    ENCODER = "enc"
    DECODER = "dec"

    @staticmethod
    def init_params():
        """Initialize CNN params"""
        print('initializing', flush=True)
        return (
            CNN.init_batch_norm(),
            CNN.init_encoders({}),
            CNN.init_decoders({}),
            CNN.init_softmax_kernels()
        )

    @staticmethod
    def create_matrix(values, row_length):
        """Create values / row_length x row_length matrix"""
        kernel = []
        row = []
        for value in values:
            row.append(value)
            if len(row) == row_length:
                kernel.append(row)
                row = []
        return kernel

    @staticmethod
    def init_kernel():
        """Initialize single kernel 7x7 """
        return(CNN.create_matrix(
            random.normal(
                0, 0.1, CNN.CONVOLUTION_KERNEL_SIZE ** 2
            ),
            CNN.CONVOLUTION_KERNEL_SIZE
        ))

    @staticmethod
    def init_kernels(
            part=None,
            structure=None,
            kernel_name=None
        ):
        """Initialize single layer kernels"""
        if kernel_name is None:
            (
                stack_number,
                layer_number,
                ch_number,
                input_number
            ) = structure
            kernel_name = (
                part +
                str(stack_number) +
                "-" +
                str(layer_number) +
                "-" +
                str(ch_number) +
                '-' +
                str(input_number)
            )
        kernel = []
        try:
            kernel = CNN.read_kernel(
                kernel_name
            )
        except FileNotFoundError:
            kernel = CNN.init_kernel()
        return (kernel_name, kernel)

    @staticmethod
    def init_layers(kernels, stack_number, part):
        """Initialize layers per stack"""
        for i in range(
                CNN.CONVOLUTION_ORDERS[part][stack_number][0]
            ): # layer
            for j in range(
                    CNN.CONVOLUTION_ORDERS[part][stack_number][1]
                ): # channel
                for k in range(
                        CNN.CONVOLUTION_ORDERS[part][stack_number][2][i]
                    ):
                    name, kernel = CNN.init_kernels(
                        part, (stack_number, i, j, k)
                    )
                    kernels[name] = kernel
        return kernels

    @staticmethod
    def init_encoders(kernels):
        """Initialize convolutional layers for SegNet particular part"""
        for i in range(0, len(CNN.CONVOLUTION_ORDERS[CNN.ENCODER])): # stack
            kernels = CNN.init_layers(kernels, i, CNN.ENCODER)
        return kernels

    @staticmethod
    def init_decoders(kernels):
        """Initialize convolutional layers for SegNet particular part"""
        for i in range(len(CNN.CONVOLUTION_ORDERS[CNN.DECODER]) - 1, -1, -1): # stack
            kernels = CNN.init_layers(kernels, i, CNN.DECODER)
        return kernels

    @staticmethod
    def init_batch_norm():
        """Initialize batch norms gamma beta."""
        batch_norm_params = {}
        params_list = ['gamma', 'beta', 'average', 'variance']
        for part in [CNN.ENCODER, CNN.DECODER]:
            side = CNN.INPUT_SIZE
            for i in range(0, len(CNN.CONVOLUTION_ORDERS[part])):
                for param_name in params_list:
                    batch_norm_params[
                        part + str(i) + "-" + param_name
                    ] = CNN.init_single_batch_norm_param(
                        part + str(i),
                        param_name,
                        CNN.CONVOLUTION_ORDERS[part][i][1] # based on number of channel output
                    )

        return batch_norm_params

    @staticmethod
    def init_softmax_kernels():
        """Initialize kernels for softmax."""
        kernels = []
        for i, part in enumerate(CNN.CONVOLUTION_ORDERS['softmax'][0]):
            kernels.append([])
            for _input in range(CNN.CONVOLUTION_ORDERS['softmax'][1]):
                _, kernel = CNN.init_kernels(
                    kernel_name='softmax-' + part + str(_input)
                )
                kernels[i].append(kernel)
        return array(kernels)

    @staticmethod
    def store_kernel(file, rows):
        """Store kernels"""
        for row in rows:
            text = ""
            for number in row:
                text += (str(number) + " ")
            file.write(text + "\n")
        return file

    @staticmethod
    def store_param(param):
        """Store params into text file"""
        for key, value in param.items():
            file = open("static/params/" + key + ".txt", "w")
            CNN.store_kernel(file, value)
            file.close()

    @staticmethod
    def read_kernel(filename):
        """Read kernel from file"""
        str_values = ""
        arr = []
        try:
            file = open("static/params/" + filename + ".txt")
            str_values = file.readlines()
            for string in str_values:
                arr.append(
                    array(string.strip().split(" "), dtype=float32)
                )
            file.close()
            return arr
        except FileNotFoundError:
            raise

    @staticmethod
    def init_single_batch_norm_param(param_structure, param_name, row_length):
        """Read batch norm params from file or initialize it directly"""
        param = None
        try:
            param = CNN.read_kernel(param_structure + "-" + param_name)
        except FileNotFoundError:
            if param_name == 'beta':
                param = zeros((row_length)) 
            elif param_name == 'gamma':
                param = ones((row_length))
            else:
                param = None
        return param

    # Forward area

    # beta and gamma should me matrix 1 x input length instead of scalar
    # tensorflow tested: tf.nn.batch_normalization(
        # a,
        # mu,
        # var,
        # tf.constant(np.zeros(len(a)), dtype=tf.float32),
        # tf.constant(np.ones(len(a)), dtype=tf.float32),
        # 0.001
    #)

    @staticmethod
    def expand_batch_norm_param(param):
        """
        Since this keep on becoming problem, each param seems has to be 
        expanded into shape (batch,channel,1,1)
        """
        param = array(param)
        return param.reshape(
            param.shape[0], param.shape[1], 1, 1
        )

    @staticmethod
    def batch_norm(
            matrices,
            beta,
            gamma,
            average=None,
            variance=None,
            epsilon=0.001
        ):
        """Calculate batch normalization from matrices in a batch"""
        matrices = array(matrices)
        beta = CNN.expand_batch_norm_param(beta)
        gamma = CNN.expand_batch_norm_param(gamma)
        if average == None:
            average = mean(matrices, axis=1)
        else:
            average = CNN.expand_batch_norm_param(average)
        if variance == None:
            variance = var(matrices, axis=1)
        else:
            variance = CNN.expand_batch_norm_param(variance)

        number_mean = matrices - average
        normalized = number_mean / (sqrt(variance + epsilon))
        scaled_shift_data = normalized * gamma + beta
        # scaled_shift_data = normalized
        print('norm: ', gamma.shape, ' ssd ', beta.shape)
        return scaled_shift_data, (
            normalized, number_mean, sqrt(variance + epsilon)
        )

    # manual tested
    @staticmethod
    def relu(matrix):
        """Process matrix with non activation ReLU function"""
        matrix = array(matrix)
        matrix[matrix <= 0] = 0
        return matrix

    # manual tested
    @staticmethod
    def max_pooling(matrix):
        """Max pooling 2 x 2 filter with stride 2"""
        result = []
        max_index = []
        for i in range(0, len(matrix) // 2):
            row = []
            row_index = []
            for j in range(0, len(matrix[i]) // 2):
                max_value = max(
                    value_with_position.ValueWithPosition(
                        matrix[i * CNN.POOLING_STRIDE][j * CNN.POOLING_STRIDE],
                        j * CNN.POOLING_STRIDE,
                        i * CNN.POOLING_STRIDE
                    ),
                    value_with_position.ValueWithPosition(
                        matrix[i * CNN.POOLING_STRIDE][j * CNN.POOLING_STRIDE + 1],
                        j * CNN.POOLING_STRIDE + 1,
                        i * CNN.POOLING_STRIDE
                    ),
                    value_with_position.ValueWithPosition(
                        matrix[i * CNN.POOLING_STRIDE + 1][j * CNN.POOLING_STRIDE],
                        j * CNN.POOLING_STRIDE,
                        i * CNN.POOLING_STRIDE + 1
                    ),
                    value_with_position.ValueWithPosition(
                        matrix[i * CNN.POOLING_STRIDE + 1][j * CNN.POOLING_STRIDE + 1],
                        j * CNN.POOLING_STRIDE + 1,
                        i * CNN.POOLING_STRIDE + 1
                    ),
                    key=lambda x: x.value
                )
                row.append(
                    max_value.value
                )
                row_index.append([max_value.x, max_value.y])
            result.append(row)
            max_index.append(row_index)
        return result, max_index

    # manual tested
    @staticmethod
    def upsampling(max_pooled, cache):
        """Create sparse matrix based on index and max pooled result"""
        sparse_matrix = zeros((cache[0], cache[1]))
        for i, row in enumerate(cache[2]):
            for j, index in enumerate(row):
                sparse_matrix[index[1]][index[0]] = max_pooled[i][j]
        return sparse_matrix

    @staticmethod
    def trainable_softmax(kernels, matrix):
        """
        Return classified matrix.
        - Kernels shape will be 77(input number)2
        - Matrix shape will be batch, width, height, channel
        """

        CNN.softmax_kernel = kernels

        matrix = forward.Forward.reverse_shape(
            tf.keras.layers.Conv2D(
                len(CNN.CONVOLUTION_ORDERS['softmax'][0]),
                CNN.CONVOLUTION_KERNEL_SIZE,
                input_shape=matrix.shape[1:],
                padding = 'same',
                kernel_initializer=CNN.get_softmax_kernel,
                use_bias=False
            )(matrix).numpy()[0]
        )

        foreground = matrix[
            CNN.CONVOLUTION_ORDERS['softmax'][0].index('fg')
        ]
        background = matrix[
            CNN.CONVOLUTION_ORDERS['softmax'][0].index('bg')
        ]

        return CNN.softmax([background, foreground]), [background, foreground]

    @staticmethod
    def get_softmax_kernel(shape, dtype=None):
        return CNN.softmax_kernel

    # tensorflow tested: tf.nn.softmax(matrix)
    @staticmethod
    def softmax(matrices):
        """Compute softmax values for each sets of scores in matrix."""
        matrices = array(matrices)
        matrices = exp (matrices - matrices.max(axis=0))
        return matrices / matrices.sum(axis=0)

    # tensorflow tested: tf.keras.losses.BinaryCrossentropy() <call>
    @staticmethod
    def cross_entropy_loss(positive_pred, negative_pred, ground_truth_matrix):
        """Return the cross entropy loss between
        ground truth and predicted one"""
        positive_pred = array(positive_pred, dtype=float32)
        negative_pred = array(negative_pred, dtype=float32)
        ground_truth_matrix = array(ground_truth_matrix, dtype=float32)
        result = - multiply(
            ground_truth_matrix,
            log(positive_pred)
        ) - multiply(
            (1 - ground_truth_matrix),
            log(negative_pred)
        )

        if mean(result) < 0:
            print('NEGATIVE LOSS')
            i = result.flatten().argmax()
            print('MAX: ', result.flatten()[i], ' AT ', i)
            print('caused by: ', positive_pred.flatten()[i], ' and ', negative_pred.flatten()[i])
            i = result.flatten().argmin()
            print('MIN: ', result.flatten()[i], ' AT ', i)
            print('caused by: ', positive_pred.flatten()[i], ' and ', negative_pred.flatten()[i])
        return mean(result)

    # Backward area

    # reference @14prakash tested
    @staticmethod
    def derivative_cross_entropy_loss(
            positive_pred,
            negative_pred,
            ground_truth_matrix
        ):
        """Return the derivative of cross entropy loss against
            softmax output"""
        positive_pred = array(positive_pred, dtype=float32)
        negative_pred = array(negative_pred, dtype=float32)
        ground_truth_matrix = array(ground_truth_matrix, dtype=float32)

        return - multiply(
            ground_truth_matrix,
            (1 / positive_pred)
        ) - multiply(
            (1 - ground_truth_matrix),
            (1 / negative_pred)
        )

    # reference @14prakash tested
    @staticmethod
    def derivative_softmax(softmax_input):
        """Return the derivative of softmax against its input"""
        background, foreground = softmax_input
        e_x_foreground = exp(foreground)
        e_x_background = exp(background)

        return [(
            (e_x_foreground * e_x_background) /
            (e_x_foreground + e_x_background) ** 2
        )] # putting inside array because there should be single channel

    # reference @14prakash tested
    @staticmethod
    def weight_gradient(convolution_input, error_result):
        """Compute weight gradient by error_result * convolution_input"""
        convolution_input = array(convolution_input, dtype=float32)
        error_result = array(error_result, dtype=float32)
        return convolve2d(
            CNN.padded(convolution_input, CNN.PADDING_SIZE),
            error_result,
            mode='valid'
        )

    # manually tested
    @staticmethod
    def padded(arr_data, padding):
        """Give padding with custom size to matrix"""
        arr_data = array(arr_data, dtype=float32)
        expanded = zeros((arr_data.shape[0] + (padding * 2), arr_data.shape[1] + (padding * 2)))
        for i, row in enumerate(arr_data):
            for j, pixel in enumerate(row):
                expanded[i + padding][j + padding] = pixel
        return expanded

    # reference @14prakash tested
    @staticmethod
    def minibatch_gradient_descent(
            current_kernel_weight,
            batch_member_weight_gradient,
            learning_rate=0.01
        ):
        """Update weight of current kernel"""
        current_kernel_weight = array(current_kernel_weight, dtype=float32)
        batch_member_weight_gradient = array(
            batch_member_weight_gradient, dtype=float32
        )
        return current_kernel_weight - learning_rate * batch_member_weight_gradient

    # reference @14prakash tested
    @staticmethod
    def derivative_convolution(before_update_kernel_weight, error_result):
        """Process convolution derivative by error_result * weight"""
        return convolve2d(
            error_result,
            flip(before_update_kernel_weight),
            mode='same'
        )

    # manually tested
    @staticmethod
    def derivative_upsampling(error_result, indices):
        """Basically it downsamples the upsampled"""
        downsampled = []
        for row in indices:
            downsampled_row = []
            for index in row:
                downsampled_row.append(error_result[int(index[1])][int(index[0])])
            downsampled.append(downsampled_row)
        return downsampled

    # tested on upsampling
    @staticmethod
    def derivative_max_pooling(error_result, cache):
        """Basically this upsamples the downsampled"""
        return CNN.upsampling(error_result, cache)

    # reference @14prakash tested
    @staticmethod
    def derivative_relu(relu_input):
        """ReLU with binary condition"""
        binary_matrix = []
        for row in relu_input:
            matrix_row = []
            for pixel in row:
                matrix_row.append(
                    1 if pixel > 0 else 0
                )
            binary_matrix.append(matrix_row)
        return binary_matrix

    @staticmethod
    def derivative_scale_shift(prev_error_result, normalized, gamma):
        """Produces error_result, gamma and beta gradient"""
        gamma_gradients = sum_array(prev_error_result * normalized, axis=0)
        print(
            'derivative scale shift shape: ',
            array(prev_error_result).shape,
            ' and ',
            array(normalized).shape,
            ' resulting gamma grad ',
            gamma_gradients.shape
        )
        return(
            prev_error_result, # beta gradient
            gamma_gradients,
            gamma_gradients * gamma # error_results
        )

    @staticmethod
    def derivative_batch_norm(error_result, cache):
        """Produces error_result of batch norm derivative"""
        _, number_mean, sqrtvar = cache

        gradient_variance = (
            0.5 *
            (
                (-1 / (sqrtvar ** 2)) *
                sum_array(error_result * number_mean, axis=0)
            ) / sqrtvar
        )

        first_part = (
            (error_result / sqrtvar) +
            (
                2 * number_mean *
                (
                    gradient_variance / len(error_result)
                )
            )
        )

        second_part = (- sum_array(first_part, axis=0) / len(error_result))

        return first_part + second_part

    @staticmethod
    def softmax_classifier(background, foreground):
        """Classify into binary values"""
        # 0 0 because naturally there will be extra dimension for batch and channel
        classified = []
        for i, row in enumerate(background):
            new_row = []
            for j, _px in enumerate(row):
                new_row.append(
                    0 if _px > foreground[i][j] else 1
                )
            classified.append(new_row)
        return classified
        