"""This script is to process data with CNN either training or testing"""

from os import path
from numpy import (
    random,
    add,
    flip,
    sqrt,
    mean,
    var,
    exp,
    array,
    multiply,
    log,
    float32
)
from scipy.signal import convolve2d
from watermarking import value_with_position

class CNN:
    """Keeps atrribute embedding map and weight"""

    CONVOLUTION_KERNEL_SIZE = 7
    POOLING_KERNEL_SIZE = 2
    POOLING_STRIDE = 2
    BATCH_SIZE = 4
    CONVOLUTION_ORDERS = [
        [2, 64],
        [2, 128],
        [3, 256],
        [3, 512],
        [3, 512]
    ]
    ENCODER = "enc"
    DECODER = "dec"

    def __init__(self, embedding_map):
        self.embedding_map = embedding_map
        self.kernels = {}
        self.batch_norm_param = {
            "beta": 0,
            "gamma": 1
        }
        #Initialize pooling indexes storage per stack for each batch member
        self.pooling_indexes = [
            [
                [None] * len(self.CONVOLUTION_ORDERS)
            ] * self.BATCH_SIZE
        ]

    def create_matrix(self, values, side):
        """Create side x side matrix"""
        kernel = []
        row = []
        for value in values:
            row.append(value)
            if len(row) == side:
                kernel.append(row)
                row = []
        return kernel

    def init_kernel(self):
        """Initialize single kernel 7x7 """
        return(self.create_matrix(
            random.normal(
                0,
                0.01,
                self.CONVOLUTION_KERNEL_SIZE ** 2
            ),
            self.CONVOLUTION_KERNEL_SIZE
        ))

    def init_kernels(self, part, stack_number, layer_number, ch_number):
        """Initialize single layer kernels"""
        kernel_name = (
            part +
            str(stack_number) +
            "-" +
            str(layer_number) +
            "-" +
            str(ch_number)
        )
        try:
            self.kernels[
                kernel_name
            ] = self.read_kernel(
                kernel_name
            )
        except FileNotFoundError:
            self.kernels[
                kernel_name
            ] = self.init_kernel()

    def init_layers(self, stack_number, part):
        """Initialize layers per stack"""
        for i in range(0, self.CONVOLUTION_ORDERS[stack_number][0]): # layer
            for j in range(0, self.CONVOLUTION_ORDERS[stack_number][1]): # channel
                self.init_kernels(part, stack_number, i, j)

    def init_encoders(self):
        """Initialize encoder convolutions. PLEASE RUN THIS FUNCTION JUST ONCE"""
        for i in range(0, len(self.CONVOLUTION_ORDERS)): # stack
            self.init_layers(i, self.ENCODER)

    def init_decoders(self):
        """Initialize decoder convolutions. PLEASE RUN THIS FUNCTION JUST ONCE"""
        for i in range(len(self.CONVOLUTION_ORDERS) - 1, -1, -1): # stack
            self.init_layers(i, self.DECODER)

    def store_kernel(self, file, rows):
        """Store kernels"""
        for row in rows:
            text = ""
            for number in row:
                text += str(number)
            file.write(text + "\n")
        return file

    def store_param(self, param, part):
        """Store params into text file"""
        for key, value in param.items():
            file = open("static/params/" + key + ".txt", "w")
            if part == 'kernel':
                self.store_kernel(file, param)
            else:
                file.write(value)
            file.close()

    def read_kernel(self, filename):
        """Read kernel from file"""
        str_values = ""
        arr = []
        try:
            file = open("static/params/" + filename + ".txt")
            str_values = file.readlines()
            for string in str_values:
                arr.append(string.strip().split(" "))
            file.close()
            return arr
        except FileNotFoundError:
            raise

    def read_batch_norm(self):
        """Read batch norm params from file"""
        try:
            file = open("static/params/beta.txt")
            self.batch_norm_param["beta"] = float(file.read())
            file.close()
            file = open("static/params/gamma.txt")
            self.batch_norm_param["gamma"] = float(file.read())
            file.close()
        except FileNotFoundError:
            pass

    # tensorflow tested: tf.nn.batch_normalization(
        # a,
        # mu,
        # var,
        # tf.constant(np.zeros(len(a)), dtype=tf.float32),
        # tf.constant(np.ones(len(a)), dtype=tf.float32),
        # 0.001
    #)
    def batch_norm(self, matrix, beta, gamma, epsilon=0.001):
        """Calculate batch normalization from single matrix"""
        average = mean(matrix)
        variance = var(matrix)
        normalized_data = []
        for number in matrix:
            normalized_data.append(
                (number - average)/(sqrt(variance + epsilon)) * gamma + beta
            )
        return normalized_data

    # manual tested
    def relu(self, matrix):
        """Process matrix with non activation ReLU function"""
        for i, row in enumerate(matrix, start=0):
            for j, value in enumerate(row, start=0):
                matrix[i][j] = max(0, value)
        return matrix

    # manual tested
    def max_pooling(self, matrix):
        """Max pooling 2 x 2 filter with stride 2"""
        result = []
        max_index = []
        for i in range(0, len(matrix) // 2):
            row = []
            for j in range(0, len(matrix[i]) // 2):
                max_value = max(
                    value_with_position.ValueWithPosition(
                        matrix[i * self.POOLING_STRIDE][j * self.POOLING_STRIDE],
                        j * self.POOLING_STRIDE,
                        i * self.POOLING_STRIDE
                    ),
                    value_with_position.ValueWithPosition(
                        matrix[i * self.POOLING_STRIDE][j * self.POOLING_STRIDE + 1],
                        j * self.POOLING_STRIDE + 1,
                        i * self.POOLING_STRIDE
                    ),
                    value_with_position.ValueWithPosition(
                        matrix[i * self.POOLING_STRIDE + 1][j * self.POOLING_STRIDE],
                        j * self.POOLING_STRIDE,
                        i * self.POOLING_STRIDE + 1
                    ),
                    value_with_position.ValueWithPosition(
                        matrix[i * self.POOLING_STRIDE + 1][j * self.POOLING_STRIDE + 1],
                        j * self.POOLING_STRIDE + 1,
                        i * self.POOLING_STRIDE + 1
                    ),
                    key=lambda x: x.value
                )
                row.append(
                    max_value.value
                )
                max_index.append([max_value.x, max_value.y])
            result.append(row)
        return result, max_index

    # tensorflow tested: tf.nn.softmax(matrix)
    def softmax(self, matrix):
        """Compute softmax values for each sets of scores in matrix."""
        e_x = exp(matrix - max(matrix))
        return e_x / e_x.sum()

    # tensorflow tested: tf.keras.losses.BinaryCrossentropy() <call>    
    def cross_entropy_loss(self, predicted_matrix, ground_truth_matrix):
        """Return the cross entropy loss between
        ground truth and predicted one"""
        predicted_matrix = array(predicted_matrix, dtype=float32)
        ground_truth_matrix = array(ground_truth_matrix, dtype=float32)
        result = - multiply(
            ground_truth_matrix,
            log(predicted_matrix)
        ) - multiply(
            (1 - ground_truth_matrix),
            log(1 - predicted_matrix)
        )
        return mean(result)

    def init_params(self):
        """Initialize CNN params"""
        self.read_batch_norm()
        self.init_encoders()
        self.init_decoders()
            # self.store_param(self.kernels)
            # self.store_param(self.batch_norm)

    def conv_per_stack(self, part, stack_number, batch):
        """Convolution as many as number of layers and channels in current stack number"""
        convolved_batch = []
        for matrix in batch:
            convolved_member = matrix
            for layer_no in range(self.CONVOLUTION_ORDERS[stack_number][0]):
                for ch_no in range(self.CONVOLUTION_ORDERS[stack_number][1]):
                    convolved_member = convolve2d(
                        convolved_member,
                        flip(
                            self.kernels[
                                part +
                                str(stack_number) +
                                "-" +
                                str(layer_no) +
                                "-" +
                                str(ch_no)
                            ]
                        ),
                        mode='same'
                    ) if ch_no == 0 else add(
                        convolve2d(
                            convolved_member,
                            flip(
                                self.kernels[
                                    part +
                                    str(stack_number) +
                                    "-" +
                                    str(layer_no) +
                                    "-" +
                                    str(ch_no)
                                ]
                            ),
                            mode='same'
                        )
                    )
            convolved_batch.append(convolved_member)

    def relu_per_batch(self, batch):
        """Process each batch member with ReLU"""
        relued_batch = []
        for matrix in batch:
            relued_batch.append(self.relu(matrix))
        return relued_batch

    def max_pooling_per_batch(self, batch, pooling_order):
        """Process each batch member with max pooling"""
        max_pooled_batch = []
        for i, matrix in enumerate(batch, start=0):
            result, max_index = self.max_pooling(matrix)
            max_pooled_batch.append(result)
            self.pooling_indexes[i][pooling_order] = max_index
        return max_pooled_batch
