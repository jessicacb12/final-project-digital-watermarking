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
    UPSAMPLING = 3
    MAX_POOLING = 4
    RELU = 5
    STANDARD_AVERAGE_INPUTS = 6

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

        # expected: 1 matrix per batch
        loss = self.per_batch_member_do(
            (self.softmax_outputs, self.ground_truth_matrices),
            self.CROSS_ENTROPY,
            len(self.softmax_outputs)
        )
        print('shape: ', loss.shape)
        print("Deriv Softmax")
        # expected: 1 matrix per batch
        loss *= self.per_batch_member_do(
            array(self.softmax_cache),
            self.SOFTMAX,
            len(self.softmax_cache)
        )
        print('shape: ', loss.shape)
        # expected: 2 weight grad/batch, 1 loss matrix/batch
        loss = self.back_softmax_convolution(loss)
        print('shape: ', loss.shape)

        for i in range(0, len(cnn.CNN.CONVOLUTION_ORDERS[cnn.CNN.DECODER])):
            print('current stack: ', i)
            print("Deriv BN update weight")
            error_result = self.batch_norm_scale_shift_get_error(
                cnn.CNN.DECODER, i, loss
            )
            print('error shape: ', array(error_result).shape)
            print("Deriv BN")
            loss = cnn.CNN.derivative_batch_norm(
                error_result,
                self.batch_norm_cache[
                    Backward.adjust_stack_number(cnn.CNN.DECODER, i)
                ]
            )
            print('shape: ', array(loss).shape)
            print("Deriv Conv")
            loss = self.convolution_update_weight_get_error(
                cnn.CNN.DECODER, i, loss
            )
            print('shape: ', array(loss).shape)
            cache = Backward.get_cache_per_stack(
                self.max_pooling_cache, i, indicesonly=True
            )
            print('cache: ', array(cache).shape)
            print("Deriv Ups")
            loss = self.per_batch_member_do(
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
        for i in range(len(cnn.CNN.CONVOLUTION_ORDERS[cnn.CNN.ENCODER]) - 1, -1, -1):
            print("Deriv Max pool")
            cache = Backward.get_cache_per_stack(
                self.max_pooling_cache, i
            )
            loss = self.per_batch_member_do(
                [
                    loss,
                    cache
                ],
                self.MAX_POOLING,
                len(cache)
            )
            print('shape: ', array(loss).shape)
            print("Deriv ReLU")
            cache = Backward.get_cache_per_stack(
                self.relu_cache, i
            )
            loss *= self.per_batch_member_do(
                cache,
                self.RELU,
                len(cache)
            )
            print('shape: ', array(loss).shape)
            print("Deriv BN update weight")
            error_result = self.batch_norm_scale_shift_get_error(
                cnn.CNN.ENCODER, i, loss
            )
            print('error shape: ', array(error_result).shape)
            print("Deriv BN")
            loss = cnn.CNN.derivative_batch_norm(
                error_result,
                self.batch_norm_cache[
                    Backward.adjust_stack_number(cnn.CNN.ENCODER, i)
                ]
            )
            print('shape: ', array(loss).shape)
            print("Deriv Conv")
            loss = self.convolution_update_weight_get_error(
                cnn.CNN.ENCODER, i, loss
            )
            print('shape: ', array(loss).shape)

        # print(self.encoder_kernels["enc0-0-0"])
        # return (
        #     self.scale_shift,
        #     self.encoder_kernels,
        #     self.decoder_kernels,
        # )

    def per_batch_member_do(
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
            elif process == self.RELU:
                result.append(
                    self.per_channel_in_batch_member_do(
                        process,
                        batch_data[i]
                    )
                )
            elif process == self.STANDARD_AVERAGE_INPUTS:
                result.append(
                    self.per_batch_member_do(
                        batch_data[i],
                        self.STANDARD_AVERAGE,
                        len(batch_data[i])
                    )
                )
            else: # for max pooling and upsampling
                result.append(
                    self.per_channel_in_batch_member_do(
                        process,
                        batch_data[1][i], # indices
                        batch_data[0][i] # error_result
                    )
                )
        if process == self.STANDARD_AVERAGE:
            result = sum_arr(result, axis=0) / len(result)
        return array(result, dtype=float32)

    def back_softmax_convolution(self, prev_error_result):
        """Update weight and get average error result for
           foreground-background convolutions"""
        mean_weight_grad = []
        loss = []

        for i, conv_input in enumerate(self.conv_softmax_cache): # per batch member
            mean_weight_grad.append(cnn.CNN.weight_gradient(
                conv_input,
                prev_error_result[i][0] # again, because there's extra channel dimension
            ))
            loss.append([
                cnn.CNN.derivative_convolution(
                    self.softmax_kernels["softmax-fg"],
                    prev_error_result[i][0]
                ),
                cnn.CNN.derivative_convolution(
                    self.softmax_kernels["softmax-bg"],
                    prev_error_result[i][0]
                )
            ])
        mean_weight_grad = self.per_batch_member_do(
            mean_weight_grad,
            self.STANDARD_AVERAGE,
            len(mean_weight_grad)
        )

        self.softmax_kernels["softmax-fg"] = self.conv_update_single_kernel(
            self.softmax_kernels["softmax-fg"], mean_weight_grad
        )
        self.softmax_kernels["softmax-bg"] = self.conv_update_single_kernel(
            self.softmax_kernels["softmax-bg"], mean_weight_grad
        )

        return array(loss, dtype=float32)

    def conv_update_single_kernel(self, kernel, weight_gradient):
        """Update convolution weights for single kernel"""
        return cnn.CNN.minibatch_gradient_descent(
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
        print('taking bn cache on ', adjusted_stack_number)
        normalized = self.batch_norm_cache[adjusted_stack_number][0]
        print('normalized shape: ', normalized.shape)
        (
            beta_gradient,
            gamma_gradient,
            error_result
        ) = cnn.CNN.derivative_scale_shift(
            prev_error_result,
            normalized,
            self.scale_shift[part + "-" + str(stack_number) + "-gamma"]
        )

        self.scale_shift[
            part + "-" + str(stack_number) + "-gamma"
        ] = cnn.CNN.minibatch_gradient_descent(
            self.scale_shift[part + "-" + str(stack_number) + "-gamma"],
            self.per_batch_member_do(
                gamma_gradient,
                self.STANDARD_AVERAGE,
                len(gamma_gradient)
            )
        )

        self.scale_shift[
            part + "-" + str(stack_number) + "-beta"
        ] = cnn.CNN.minibatch_gradient_descent(
            self.scale_shift[part + "-" + str(stack_number) + "-beta"],
            self.per_batch_member_do(
                beta_gradient,
                self.STANDARD_AVERAGE,
                beta_gradient.shape[0]
            )
        )

        return error_result

    @staticmethod
    def adjust_stack_number(part, stack_number):
        """Adjustment because of cache from behind. Only used if
           process exists both in encoder and decoder"""
        return stack_number + 1 if part == cnn.CNN.ENCODER else (
            len(cnn.CNN.CONVOLUTION_ORDERS[cnn.CNN.ENCODER]) * 2 - stack_number
        )

    def convolution_update_weight_get_error(self, part, stack_number, error_result):
        """Update convolution weights and return error_result per stack"""
        for i in range(0, cnn.CNN.CONVOLUTION_ORDERS[part][stack_number][0]): # layer
            error_result = self.convolution_update_weight_per_layer(
                part,
                stack_number,
                i,
                error_result
            )
        return error_result

    def convolution_update_weight_per_layer(
            self,
            part,
            stack_number,
            layer,
            error_result
        ):
        """Update convolution weights and return error per layer"""
        weight_gradient = self.weight_gradient_per_layer(
            part, stack_number, layer, error_result
        )
        print('weight grad: ', array(weight_gradient).shape)

        error_per_layer = [[] for _ in range(0, len(error_result))]
        kernel = self.encoder_kernels if part == cnn.CNN.ENCODER else self.decoder_kernels
        for i in range(0, cnn.CNN.CONVOLUTION_ORDERS[part][stack_number][1]): # channel
            name = part + str(stack_number) + "-" + str(layer) + "-" + str(i)
            for j, batch_member in enumerate(error_result): # batch in error result
                error_per_layer[j].append(
                    cnn.CNN.derivative_convolution(
                        kernel[name],
                        batch_member[i]
                    )
                )
            kernel[name] = self.conv_update_single_kernel(
                kernel[name], weight_gradient[i]
            )
        return array(error_per_layer)

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
        for j in range(0, len(error_result[0])): # per channel in error result
            weight_gradients_per_batch = []
            for i, batch_member_input in enumerate(self.convolution_cache):
                weight_gradients_per_member = []
                # if len(batch_member_input[part][stack_number]) > 0:
                for single_input in batch_member_input[
                        part
                    ][
                        stack_number
                    ][
                        layer_number
                    ]: # per channel of input
                    weight_gradients_per_member.append(
                        cnn.CNN.weight_gradient(
                            single_input,
                            error_result[i][j]
                        )
                    )
                weight_gradients_per_batch.append(
                    sum_arr(weight_gradients_per_member, axis=0)
                )
            weight_gradients.append(
                self.per_batch_member_do(
                    weight_gradients_per_batch,
                    self.STANDARD_AVERAGE,
                    len(weight_gradients_per_batch)
                )
            )
        return weight_gradients

    @staticmethod
    def get_cache_per_stack(
            batch_cache,
            stack_number,
            indicesonly=False
        ):
        """Get cache for particular stack"""
        taken_cache = []
        for batch_member in batch_cache:
            cache_per_batch = []
            if len(batch_member) > 0:
                inputs = batch_member[stack_number]
                print('original structure: ', array(inputs).shape)
                for single_input in inputs:
                    if indicesonly:
                        cache_per_batch.append(single_input[2])
                    else:
                        cache_per_batch = inputs
                        break
            taken_cache.append(cache_per_batch)
        return taken_cache

    def per_channel_in_batch_member_do(self, process, cache, loss=None):
        """Process per channel in batch member"""
        summed_loss = sum_arr(loss, axis=0)
        per_batch_member_result = []
        for channel in cache:
            if process == self.UPSAMPLING:
                per_batch_member_result.append(
                    cnn.CNN.derivative_upsampling(
                        summed_loss,
                        channel
                    )
                )
            elif process == self.MAX_POOLING:
                per_batch_member_result.append(
                    cnn.CNN.derivative_max_pooling(
                        summed_loss,
                        channel
                    )
                )
            elif process == self.RELU:
                per_batch_member_result.append(
                    cnn.CNN.derivative_relu(channel)
                )
        return per_batch_member_result
