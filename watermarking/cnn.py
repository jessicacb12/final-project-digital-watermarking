"""This script is to process data with CNN either training or testing"""

from numpy import random
from os import path

class CNN:
    """Keeps atrribute embedding map and weight"""

    CONVOLUTION_KERNEL_SIZE = 7
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
        self.batch_norm = {
            "beta": 0,
            "gamma": 1
        }

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

    def init_kernels(self, type, stack_number, layer_number):
        """Initialize single layer kernels"""
        kernel_name = type + str(stack_number) + "-" + str(layer_number)
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

    def init_layers(self, stack_number, type):
        """Initialize layers per stack"""
        for i in range(0, self.CONVOLUTION_ORDERS[stack_number][0]): # layer
            for j in range(0, self.CONVOLUTION_ORDERS[stack_number][1]): # channel
                self.init_kernels(type, stack_number, i)

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

    def store_param(self, param, type):
        """Store params into text file"""
        for key, value in param.items():
            file = open("static/params/" + key + ".txt", "w")
            if type == 'kernel':
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
            self.batch_norm["beta"] = float(file.read())
            file.close()
            file = open("static/params/gamma.txt")
            self.batch_norm["gamma"] = float(file.read())
            file.close()
        except FileNotFoundError:
            pass

    def init_params(self):
        """Initialize CNN params"""
        self.read_batch_norm()
        self.init_encoders()
        self.init_decoders()
            # self.store_param(self.kernels)
            # self.store_param(self.batch_norm)
