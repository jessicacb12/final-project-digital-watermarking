"""This script is to do backpropagation in CNN per batch"""

from numpy import array, float32
from numpy import nansum as sum_arr
from watermarking import cnn, forward

class Backward:
    """All derivative functions. Returns updated params"""

    softmax_outputs = []
    ground_truth_matrices = []

    convolution_cache = [] # checked, right structure
    batch_norm_cache = []
    relu_cache = []
    max_pooling_cache = []
    conv_softmax_cache = []
    softmax_cache = []

    # to be updated
    encoder_kernels = {}
    decoder_kernels = {}
    scale_shift = {}
    softmax_kernels = {}

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
            self.conv_softmax_cache,
            self.softmax_cache
        ) = cache
        (
            self.scale_shift,
            self.encoder_kernels,
            self.decoder_kernels,
            self.softmax_kernels
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
        print('shape: ', array(loss).shape)
        print("Deriv Softmax")
        loss *= self.average_result_for_process(
            self.softmax_cache,
            self.SOFTMAX,
            len(self.softmax_cache)
        )
        print('shape: ', array(loss).shape)
        loss *= self.back_softmax_convolution(loss)
        print('shape: ', array(loss).shape)

        for i in range(0, len(cnn.CNN.CONVOLUTION_ORDERS[cnn.CNN.DECODER])):
            print('current stack: ', i)
            print("Deriv BN update weight")
            error_result = self.batch_norm_scale_shift_get_error(
                cnn.CNN.DECODER, i, loss
            )
            print('error shape: ', array(error_result).shape)
            print("Deriv BN")
            cache = Backward.get_cache_per_stack(
                self.batch_norm_cache,
                Backward.adjust_stack_number(cnn.CNN.DECODER, i)
            )
            loss *= self.average_result_for_process(
                (error_result, cache),
                self.BATCH_NORM,
                cache.shape[0]
            )
            print('shape: ', array(loss).shape)
            print("Deriv Conv")
            loss *= self.convolution_update_weight_get_error(
                cnn.CNN.DECODER, i, loss
            )
            print('shape: ', array(loss).shape)
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
            print('shape: ', array(loss).shape)

        # # encoder
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
            error_result = self.batch_norm_scale_shift_get_error(
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

        # print(self.encoder_kernels["enc0-0-0"])
        # return (
        #     self.scale_shift,
        #     self.encoder_kernels,
        #     self.decoder_kernels,
        # )

    def average_result_for_process(
            self,
            batch_data,
            process,
            length
        ):
        """Return average result for batch process"""
        result = []
        for i in range(0, length): # per batch/defined length
            if process == self.CROSS_ENTROPY:
                result.append(cnn.CNN.derivative_cross_entropy_loss(
                    batch_data[0][i][0], # positive softmax
                    batch_data[0][i][1], # negative softmax
                    batch_data[1][i] # ground_truth
                ))
            elif process == self.SOFTMAX:
                result.append(cnn.CNN.derivative_softmax(
                    batch_data[i] # softmax input
                ))
            elif process == self.STANDARD_AVERAGE:
                result.append(batch_data[i])
            elif process == self.BATCH_NORM:
                print(batch_data[1][i].shape)
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
        total = sum_arr(result, axis=0) / length
        return total[0] if len(total.shape) > 2 else total

    def back_softmax_convolution(self, prev_error_result):
        """Update weight and get average error result for
           foreground-background convolutions"""
        mean_weight_grad = []
        for conv_input in self.conv_softmax_cache: # per batch member
            mean_weight_grad.append(cnn.CNN.weight_gradient(
                conv_input, prev_error_result
            ))
        mean_weight_grad = self.average_result_for_process(
            mean_weight_grad,
            self.STANDARD_AVERAGE,
            len(mean_weight_grad)
        )

        fg_kernel, self.softmax_kernels["softmax-fg"] = self.conv_update_single_kernel(
            self.softmax_kernels["softmax-fg"], mean_weight_grad
        )
        bg_kernel, self.softmax_kernels["softmax-bg"] = self.conv_update_single_kernel(
            self.softmax_kernels["softmax-bg"], mean_weight_grad
        )

        return self.average_result_for_process(
            [
                cnn.CNN.derivative_convolution(fg_kernel, prev_error_result),
                cnn.CNN.derivative_convolution(bg_kernel, prev_error_result)
            ], self.STANDARD_AVERAGE, 2
        )

    def conv_update_single_kernel(self, kernel, weight_gradient):
        """Update convolution weights for single kernel"""
        return kernel, cnn.CNN.minibatch_gradient_descent(
            kernel,
            weight_gradient
        )

    def batch_norm_scale_shift_get_error(
            self,
            part,
            stack_number,
            prev_error_result
        ):
        """Scale shift batch norm weights in particular stack
           then return batch norm error result"""
        adjusted_stack_number = Backward.adjust_stack_number(part, stack_number)
        normalized = Backward.get_cache_per_stack(
            self.batch_norm_cache,
            adjusted_stack_number,
            indicesonly=False,
            normalizedonly=True
        )
        print('normalized shape: ', normalized.shape)
        (
            beta_gradient,
            gamma_gradient,
            error_result
        ) = cnn.CNN.derivative_scale_shift(
            prev_error_result,
            self.average_result_for_process(
                normalized,
                self.STANDARD_AVERAGE,
                normalized.shape[0]
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
            len(cnn.CNN.CONVOLUTION_ORDERS[cnn.CNN.ENCODER]) * 2 - stack_number - 1
        )

    def convolution_update_weight_get_error(self, part, stack_number, error_result):
        """Update convolution weights and return error_result per stack"""
        for i in range(0, cnn.CNN.CONVOLUTION_ORDERS[part][stack_number][0]): # layer
            weight_gradient = self.weight_gradient_per_layer(
                part, stack_number, i, error_result
            )
            error_result = self.convolution_update_weight_per_layer(
                part,
                stack_number,
                i,
                weight_gradient,
                error_result
            )
        return error_result

    def convolution_update_weight_per_layer(
            self,
            part,
            stack_number,
            layer,
            weight_gradient,
            error_result
        ):
        """Update convolution weights and return error per layer"""
        error_per_layer = []
        original_kernel = None
        kernel = self.encoder_kernels if part == cnn.CNN.ENCODER else self.decoder_kernels
        for i in range(0, cnn.CNN.CONVOLUTION_ORDERS[part][stack_number][1]): # channel
            name = part + str(stack_number) + "-" + str(layer) + "-" + str(i)
            original_kernel, kernel[name] = self.conv_update_single_kernel(
                kernel[name], weight_gradient
            )
            error_per_layer.append(
                cnn.CNN.derivative_convolution(
                    original_kernel,
                    error_result
                )
            )
        return self.average_result_for_process(
            error_per_layer,
            self.STANDARD_AVERAGE,
            len(error_per_layer)
        )

    def weight_gradient_per_layer(
            self,
            part,
            stack_number,
            layer_number,
            error_result
        ):
        """Return averaged weight gradient per stack"""
        weight_gradients = []
        part = forward.Forward.ENCODER if part == cnn.CNN.ENCODER else forward.Forward.DECODER
        for batch_member_input in self.convolution_cache:
            if len(batch_member_input[part][stack_number]) > 0:
                weight_gradients_per_channel = []
                for single_input in batch_member_input[
                        part
                    ][
                        stack_number
                    ][
                        layer_number
                    ]: # per channel of input
                    weight_gradients_per_channel.append(
                        cnn.CNN.weight_gradient(
                            single_input,
                            error_result
                        )
                    )
                weight_gradients.append(
                    self.average_result_for_process(
                        weight_gradients_per_channel,
                        self.STANDARD_AVERAGE,
                        len(weight_gradients_per_channel)
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
                inputs = batch_member[stack_number]
                print('original structure: ', array(inputs).shape)
                for single_input in inputs:
                    if normalizedonly:
                        cache_per_batch.append(single_input[0]) # 0 is norm
                    if indicesonly:
                        cache_per_batch.append(single_input[2])
        return array(cache_per_batch, dtype=float32)
