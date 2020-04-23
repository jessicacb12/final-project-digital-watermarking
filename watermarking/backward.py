"""This script is to do backpropagation in CNN per batch"""

from copy import deepcopy
from numpy import array, float32
from numpy import sum as sum_arr
from watermarking import cnn, forward

class Backward:
    """All derivative functions. Returns updated params"""

    softmax_outputs = []
    ground_truth_matrices = []

    convolution_cache = [] # checked, right structure
    batch_norm_cache = []
    relu_cache = []
    max_pooling_cache = []
    softmax_cache = []

    # to be updated
    encoder_kernels = {}
    decoder_kernels = {}
    scale_shift = {}

    # process code
    CROSS_ENTROPY = 0
    SOFTMAX = 1
    STANDARD_AVERAGE = 2
    BATCH_NORM = 3
    UPSAMPLING = 4
    MAX_POOLING = 5
    RELU = 6

    def __init__(
            self,
            softmax_outputs,
            cache,
            params,
            ground_truth_matrices
        ):
        self.softmax_outputs = softmax_outputs
        (
            self.convolution_cache,
            self.batch_norm_cache,
            self.relu_cache,
            self.max_pooling_cache,
            self.softmax_cache
        ) = cache
        (
            self.scale_shift,
            self.encoder_kernels,
            self.decoder_kernels,
        ) = params
        self.ground_truth_matrices = ground_truth_matrices

    def run(self):
        """Function to be called from other classes"""
        # decoder
        print("Decoder")
        # print(array(self.convolution_cache[0][0][0]).shape)
        print("Deriv Cross Entropy")
        loss = self.average_result_for_process(
            (self.softmax_outputs, self.ground_truth_matrices),
            self.CROSS_ENTROPY,
            len(self.softmax_outputs)
        )
        print("Deriv Softmax")
        loss *= self.average_result_for_process(
            self.softmax_cache,
            self.SOFTMAX,
            len(self.softmax_cache)
        )

        for i in range(0, len(cnn.CNN.CONVOLUTION_ORDERS)):
            print('current stack: ', i)
            print("Deriv BN update weight")
            error_result = self.batch_norm_update_weight_get_error(
                cnn.CNN.DECODER, i, loss
            )
            print("Deriv BN")
            cache = Backward.get_cache_per_stack(
                self.batch_norm_cache,
                Backward.adjust_stack_number(cnn.CNN.DECODER, i)
            )
            loss *= self.average_result_for_process(
                (error_result, cache),
                self.BATCH_NORM,
                len(cache)
            )
            print("Deriv Conv")
            loss *= self.convolution_update_weight_get_error(
                cnn.CNN.DECODER, i, loss
            )
            cache = Backward.get_cache_per_stack(
                self.max_pooling_cache, i, indicesonly=True
            )
            print("Deriv Ups")
            loss = self.average_result_for_process(
                [
                    loss,
                    cache
                ],
                self.UPSAMPLING,
                len(cache)
            )

        # encoder
        print("Encoder")
        for i in range(len(cnn.CNN.CONVOLUTION_ORDERS) - 1, -1, -1):
            print("Deriv Max pool")
            cache = Backward.get_cache_per_stack(
                self.max_pooling_cache, i
            )
            loss = self.average_result_for_process(
                [
                    loss,
                    cache
                ],
                self.MAX_POOLING,
                len(cache)
            )
            print("Deriv ReLU")
            cache = Backward.get_cache_per_stack(
                self.relu_cache, i
            )
            loss *= self.average_result_for_process(
                cache,
                self.RELU,
                len(cache)
            )
            print("Deriv BN update weight")
            error_result = self.batch_norm_update_weight_get_error(
                cnn.CNN.ENCODER, i, loss
            )
            print("Deriv BN")
            cache = Backward.get_cache_per_stack(
                self.batch_norm_cache, i
            )
            loss *= self.average_result_for_process(
                (error_result, cache),
                self.BATCH_NORM,
                len(cache)
            )
            print("Deriv Conv")
            loss *= self.convolution_update_weight_get_error(
                cnn.CNN.ENCODER, i, loss
            )

        print(self.encoder_kernels["enc0-0-0"])
        return (
            self.scale_shift,
            self.encoder_kernels,
            self.decoder_kernels,
        )

    def average_result_for_process(
            self,
            batch_data,
            process,
            length
        ):
        """Return average result for batch process"""
        result = []
        for i in range(0, length):
            if process == self.CROSS_ENTROPY:
                result.append(cnn.CNN.derivative_cross_entropy_loss(
                    batch_data[0][i], # softmax output
                    batch_data[1][i] # ground_truth
                ))
            elif process == self.SOFTMAX:
                result.append(cnn.CNN.derivative_softmax(
                    batch_data[i] # softmax input
                ))
            elif process == self.STANDARD_AVERAGE:
                result.append(batch_data[i])
            elif process == self.BATCH_NORM:
                result.append(
                    cnn.CNN.derivative_batch_norm(
                        batch_data[0], # error result
                        batch_data[1][i] # cache
                    )
                )
            elif process == self.UPSAMPLING:
                result.append(
                    cnn.CNN.derivative_upsampling(
                        batch_data[0], # error_result
                        batch_data[1][i] # indices
                    )
                )
            elif process == self.MAX_POOLING:
                result.append(
                    cnn.CNN.derivative_max_pooling(
                        batch_data[0], # error_result
                        batch_data[1][i] # indices
                    )
                )
            elif process == self.RELU:
                result.append(
                    cnn.CNN.derivative_relu(
                        batch_data[i]
                    )
                )
        return sum_arr(result, axis=0) / length

    def batch_norm_update_weight_get_error(
            self,
            part,
            stack_number,
            prev_error_result
        ):
        """Scale shift batch norm weights in particular stack
           then return batch norm error result"""
        normalized = Backward.get_cache_per_stack(
            self.batch_norm_cache,
            stack_number,
            indicesonly=False,
            normalizedonly=True
        )
        (
            beta_gradient,
            gamma_gradient,
            error_result
        ) = cnn.CNN.derivative_scale_shift(
            prev_error_result,
            self.average_result_for_process(
                normalized,
                self.STANDARD_AVERAGE,
                len(normalized)
            ),
            self.scale_shift[part + "-" + str(stack_number) + "-gamma"]
        )

        self.scale_shift[
            part + "-" + str(stack_number) + "-gamma"
        ] = cnn.CNN.minibatch_gradient_descent(
            self.scale_shift[part + "-" + str(stack_number) + "-gamma"],
            gamma_gradient
        )

        self.scale_shift[
            part + "-" + str(stack_number) + "-beta"
        ] = cnn.CNN.minibatch_gradient_descent(
            self.scale_shift[part + "-" + str(stack_number) + "-beta"],
            beta_gradient
        )

        return error_result

    @staticmethod
    def adjust_stack_number(part, stack_number):
        """Adjustment because of cache from behind. Only used if
           process exists both in encoder and decoder"""
        return stack_number if part == cnn.CNN.ENCODER else (
            len(cnn.CNN.CONVOLUTION_ORDERS) * 2 - stack_number - 1
        )

    def convolution_update_weight_get_error(self, part, stack_number, error_result):
        """Update convolution weights and return error_result per stack"""
        for i in range(0, cnn.CNN.CONVOLUTION_ORDERS[stack_number][0]): # layer
            error_per_layer = []
            weight_gradient = self.weight_gradient_per_layer(
                part, stack_number, i, error_result
            )
            for j in range(0, cnn.CNN.CONVOLUTION_ORDERS[stack_number][1]): # channel
                kernel = self.encoder_kernels[
                    part + str(stack_number) + "-" +
                    str(i) + "-" + str(j)
                ] if part == cnn.CNN.ENCODER else self.decoder_kernels[
                    part + str(stack_number) + "-" +
                    str(i) + "-" + str(j)
                ]
                original_kernel = deepcopy(kernel)
                kernel = cnn.CNN.minibatch_gradient_descent(
                    kernel,
                    weight_gradient
                )
                error_per_layer.append(
                    cnn.CNN.derivative_convolution(
                        original_kernel,
                        error_result
                    )
                )
            error_result = self.average_result_for_process(
                error_per_layer,
                self.STANDARD_AVERAGE,
                len(error_per_layer)
            )
        return error_result

    def weight_gradient_per_layer(self, part, stack_number, layer_number, error_result):
        """Return averaged weight gradient per stack"""
        weight_gradients = []
        part = forward.Forward.ENCODER if part == cnn.CNN.ENCODER else forward.Forward.DECODER
        for batch_member_input in self.convolution_cache:
            # print(batch_member_input)
            if len(batch_member_input[part][stack_number]) > 0:
                weight_gradients.append(
                    cnn.CNN.weight_gradient(
                        batch_member_input[part][stack_number][layer_number],
                        error_result
                    )
                )
        return self.average_result_for_process(
            weight_gradients,
            self.STANDARD_AVERAGE,
            len(weight_gradients)
        )

    @staticmethod
    def get_cache_per_stack(
            batch_cache,
            stack_number,
            indicesonly=False,
            normalizedonly=False
        ):
        """Get cache for particular stack"""
        cache_per_batch = []
        for batch_member in batch_cache:
            if len(batch_member) > 0:
                cache = batch_member[stack_number]
                if indicesonly:
                    cache = cache[2]
                elif normalizedonly:
                    cache = cache[0]
                cache_per_batch.append(cache)
        return cache_per_batch
        