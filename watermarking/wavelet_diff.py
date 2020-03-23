"""This script is to encapsulate wavelet diff and its position as an object."""
from watermarking import value_with_position

class WaveletDiff(value_with_position.ValueWithPosition):
    """Only contains static method."""
    def __init__(self, value, x, y):
        super(self)

    @staticmethod
    def get_array_of_values_from(wavelet_diffs):
        """This method is used to get median from wavelet diffs."""
        """numpy.median is not supported to get median from objects :\."""
        arr = []
        for diff in wavelet_diffs:
            arr.append(diff.value)
        return arr
        