"""This script is used in embedding watermark to host image."""

from json import dumps
from pywt import dwt2, idwt2
from numpy import median
from tifffile import imsave
from cv2 import cvtColor, COLOR_BGR2RGB, imread
from watermarking import wavelet_diff
from watermarking import process

class Embedding:
    """Produces watermarked image with key attached"""

    RED = 0
    GREEN = 1
    BLUE = 2
    FILENAME = "data/watermarked_result"

    def get_color_at(self, color, image, col, row):
        """Return single color channel of pixel."""
        return image[row][col][color]

    def get_single_color_image(self, color, image):
        """Return single color channel of image."""
        single_color_image = []

        for row in range(0, len(image) - 1):
            colors = []
            for col in range(0, len(image[row]) - 1):
                colors.append(self.get_color_at(color, image, col, row))
            single_color_image.append(colors)
        return single_color_image

    def get_wave_diff(self, horizontal, vertical):
        """Get wavelet difference from horizontal and vertical."""
        return abs(horizontal - vertical)

    def get_descending_wavelet_diffs(self, horizontals, verticals, num_of_result):
        """Return <number of desired results> wavelet diffs from horizontal and vertical."""
        wavelet_diffs = []
        for i in range(0, len(horizontals)):
            for j in range(0, len(horizontals)):
                wavelet_diffs.append(
                    wavelet_diff.WaveletDiff(
                        self.get_wave_diff(horizontals[i][j], verticals[i][j]),
                        j,
                        i
                    )
                )
        descended = sorted(wavelet_diffs, key=lambda x: x.value, reverse=True)
        return descended[0:num_of_result]

    def create_key(self, wavelet_diffs, px_per_row):
        """Create key to get watermark from host on extraction."""
        text = ''
        current_row_length = 1
        for data in wavelet_diffs:
            text += ("(" + str(data.x) + "," + str(data.y) + ")")
            if current_row_length == px_per_row:
                text += "\n"
                current_row_length = 1
            else:
                current_row_length += 1
        return text

    def get_v_value(self, wavelet_diffs):
        """get v value based on median."""
        return median(wavelet_diff.WaveletDiff.get_array_of_values_from(wavelet_diffs))

    def get_thresholds(self, v_value, embedding_strength=40):
        """get threshold for watermark bit 0 and 1"""
        return (v_value - embedding_strength/2), (v_value + embedding_strength/2)

    def get_x_array(self, threshold0, threshold1, wavelet_diffs):
        """Get x array or adjustment quantity that will be used to represent watermark."""
        payload0 = []
        payload1 = []
        for diff in wavelet_diffs:
            payload0.append(diff.value - threshold0)
            payload1.append(threshold1 - diff.value)
        return payload0, payload1

    def embed_0(self, diff, threshold0, horizontal, vertical, payload0):
        """Embed watermark bit 0 to horizontal and vertical."""
        if diff > threshold0:
            if horizontal >= vertical:
                horizontal = horizontal - payload0 / 2
                vertical = vertical + payload0 / 2
            else:
                horizontal = horizontal + payload0 / 2
                vertical = vertical - payload0 / 2
        return horizontal, vertical

    def embed_1(self, diff, threshold1, horizontal, vertical, payload1):
        """Embed watermark bit 1 to horizontal and vertical."""
        if diff < threshold1:
            if horizontal >= vertical:
                horizontal = horizontal + payload1 / 2
                vertical = vertical - payload1 / 2
            else:
                horizontal = horizontal - payload1 / 2
                vertical = vertical + payload1 / 2
        return horizontal, vertical

    def embed_based_on_wavelet_diffs(self, wavelet_diffs, horizontal, vertical, watermark):
        """Return watermark-embedded horizontal and vertical."""
        # 3c - i. Get threshold to search for X array and to embed later using v value
        threshold0, threshold1 = self.get_thresholds(self.get_v_value(wavelet_diffs))

        # 3c - ii. Get adjustment quantity (x array)
        payload0, payload1 = self.get_x_array(threshold0, threshold1, wavelet_diffs)

        k = 0
        for i in range(0, len(watermark)):
            for j in range(0, len(watermark[0])):
                y = wavelet_diffs[k].y
                x = wavelet_diffs[k].x

                horizontal[y][x], vertical[y][x] = self.embed_0(
                    wavelet_diffs[k].value,
                    threshold0,
                    horizontal[y][x],
                    vertical[y][x],
                    payload0[k]
                ) if watermark[i][j] == 0 else self.embed_1(
                    wavelet_diffs[k].value,
                    threshold1,
                    horizontal[y][x],
                    vertical[y][x],
                    payload1[k]
                )
                k += 1
        return horizontal, vertical

    def embed_to_subbands(self, watermark, horizontal, vertical):
        """Return created key and watermark-embedded horizontal and vertical."""
        wavelet_diffs = self.get_descending_wavelet_diffs(
            horizontal,
            vertical,
            (len(watermark) ** 2)
        )

        key = self.create_key(wavelet_diffs, len(watermark))

        horizontal, vertical = self.embed_based_on_wavelet_diffs(
            wavelet_diffs,
            horizontal,
            vertical,
            watermark
        )

        return horizontal, vertical, key

    def put_back_color_in_image(self, color, single_color_image, ori_image):
        """Put back watermark embedded color channel to image"""
        for row in range(0, len(ori_image)):
            for col in range(0, len(ori_image[row])):
                ori_image[row][col][color] = single_color_image[row][col]
        return ori_image

    def save_tiff(self, image, str_key):
        """Save both image and key to tiff file."""
        imsave(
            process.Process.ROOT + self.FILENAME + ".tif",
            image,
            description=dumps(dict(
                key=str_key
            ))
        )

    def embed_watermark(self, host, watermark):
        """Function that will be called to embed watermark. Return numpy array."""
        to_be_embedded = self.get_single_color_image(self.BLUE, host)

        main, (vertical, horizontal, diagonal) = dwt2(to_be_embedded, 'haar')
        horizontal, vertical, key = self.embed_to_subbands(watermark, horizontal, vertical)
        embedded = idwt2((main, (vertical, horizontal, diagonal)), 'haar')

        watermarked = self.put_back_color_in_image(self.BLUE, embedded, host)

        # because openCV flips the order of RGB to BGR
        self.save_tiff(cvtColor(watermarked, COLOR_BGR2RGB), key)
        return imread(process.Process.ROOT + self.FILENAME + ".tif")
        