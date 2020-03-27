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
    float32
)
from numpy import max as max_from_array
from watermarking import value_with_position

class CNN:
    """Keeps atrribute embedding map and weight"""

    CONVOLUTION_KERNEL_SIZE = 7
    POOLING_KERNEL_SIZE = 2
    POOLING_STRIDE = 2
    CONVOLUTION_ORDERS = [
        [2, 64],
        [2, 128],
        [3, 256],
        [3, 512],
        [3, 512]
    ]
    ENCODER = "enc"
    DECODER = "dec"

    def __init__(
            self,
            embedding_map=None,
            training_images=None,
            istraining=False
        ):
        self.istraining = istraining
        self.training_images = training_images
        self.embedding_map = embedding_map

    @staticmethod
    def create_matrix(values, side):
        """Create side x side matrix"""
        kernel = []
        row = []
        for value in values:
            row.append(value)
            if len(row) == side:
                kernel.append(row)
                row = []
        return kernel

    @staticmethod
    def init_kernel():
        """Initialize single kernel 7x7 """
        return(CNN.create_matrix(
            random.normal(
                0,
                0.01,
                CNN.CONVOLUTION_KERNEL_SIZE ** 2
            ),
            CNN.CONVOLUTION_KERNEL_SIZE
        ))

    @staticmethod
    def init_kernels(part, stack_number, layer_number, ch_number):
        """Initialize single layer kernels"""
        kernel_name = (
            part +
            str(stack_number) +
            "-" +
            str(layer_number) +
            "-" +
            str(ch_number)
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
        for i in range(0, CNN.CONVOLUTION_ORDERS[stack_number][0]): # layer
            for j in range(0, CNN.CONVOLUTION_ORDERS[stack_number][1]): # channel
                name, kernel = CNN.init_kernels(part, stack_number, i, j)
                kernels[name] = kernel
        return kernels

    @staticmethod
    def init_encoders(kernels):
        """Initialize encoder convolutions. PLEASE RUN THIS FUNCTION JUST ONCE"""
        for i in range(0, len(CNN.CONVOLUTION_ORDERS)): # stack
            kernels = CNN.init_layers(kernels, i, CNN.ENCODER)
        return kernels

    @staticmethod
    def init_decoders(kernels):
        """Initialize decoder convolutions. PLEASE RUN THIS FUNCTION JUST ONCE"""
        for i in range(len(CNN.CONVOLUTION_ORDERS) - 1, -1, -1): # stack
            kernels = CNN.init_layers(kernels, i, CNN.DECODER)
        return kernels

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
    def store_param(param, part):
        """Store params into text file"""
        for key, value in param.items():
            file = open("static/params/" + key + ".txt", "w")
            if part == 'kernel':
                CNN.store_kernel(file, value)
            else:
                file.write(str(value))
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
    def read_batch_norm(batch_norm_param):
        """Read batch norm params from file"""
        try:
            file = open("static/params/beta.txt")
            batch_norm_param["beta"] = float(file.read())
            file.close()
            file = open("static/params/gamma.txt")
            batch_norm_param["gamma"] = float(file.read())
            file.close()
        except FileNotFoundError:
            pass
        return batch_norm_param

    # tensorflow tested: tf.nn.batch_normalization(
        # a,
        # mu,
        # var,
        # tf.constant(np.zeros(len(a)), dtype=tf.float32),
        # tf.constant(np.ones(len(a)), dtype=tf.float32),
        # 0.001
    #)
    @staticmethod
    def batch_norm(matrix, beta, gamma, epsilon=0.001):
        """Calculate batch normalization from single matrix"""
        average = mean(matrix)
        variance = var(matrix)

        # number_mean_arr = []
        # normalized_data = []
        # scaled_shift_data = []

        number_mean = matrix - average
        normalized = number_mean/(sqrt(variance + epsilon))
        scaled_shift_data = normalized * gamma + beta

        # for number in matrix:
        #     number_mean = number - average
        #     number_mean_arr.append(number_mean)
        #     normalized = number_mean/(sqrt(variance + epsilon))
        #     normalized_data.append(normalized)
        #     scaled_shift_data.append(
        #         normalized * gamma + beta
        #     )
        return scaled_shift_data, (
            normalized, number_mean, variance, epsilon
        )

    # manual tested
    @staticmethod
    def relu(matrix):
        """Process matrix with non activation ReLU function"""
        for i, row in enumerate(matrix, start=0):
            for j, value in enumerate(row, start=0):
                matrix[i][j] = max(0, value)
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

    @staticmethod
    def upsampling(max_pooled, cache):
        """Create sparse matrix based on index and max pooled result"""
        sparse_matrix = [[0] * cache[1]] * cache[0]
        for i, row in enumerate(cache[2]):
            for j, index in enumerate(row):
                sparse_matrix[index[1]][index[0]] = max_pooled[i][j]
        return sparse_matrix

    # tensorflow tested: tf.nn.softmax(matrix)
    @staticmethod
    def softmax(matrix):
        """Compute softmax values for each sets of scores in matrix."""
        e_x = exp(matrix - max_from_array(matrix))
        return e_x / e_x.sum()

    # tensorflow tested: tf.keras.losses.BinaryCrossentropy() <call>
    @staticmethod
    def cross_entropy_loss(predicted_matrix, ground_truth_matrix):
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

    @staticmethod
    def init_params():
        """Initialize CNN params"""
        print('initializing', flush=True)
        batch_norm = {'beta' : 0, 'gamma' : 1}
        batch_norm = CNN.read_batch_norm(batch_norm)
        return (
            batch_norm,
            CNN.init_encoders({}),
            CNN.init_decoders({})
        )
    # def run(self):
    #     """Run CNN either it's training or testing"""
    #     # if self.istraining:
    #     #     self.training()
             