"""This script is used in embedding watermark to host image."""

from copy import deepcopy
from json import dumps
from pywt import dwt2, idwt2
from numpy import sum as sum_arr
from numpy import mean, median, std, dot, uint8
from numpy.linalg import norm
from PIL.Image import fromarray
from PIL.TiffImagePlugin import ImageFileDirectory
from cv2 import cvtColor, COLOR_BGR2RGB, imread
from watermarking import wavelet_diff
from watermarking import process

class Embedding:
    """Produces watermarked image with key attached"""

    RED = 0
    GREEN = 1
    BLUE = 2
    FILENAME = "data/watermarked_result"
    KEY_LOCATION = 37000

    @staticmethod
    def get_color_at(color, image, col, row):
        """Return single color channel of pixel."""
        return image[row][col][color]

    @staticmethod
    def get_single_color_image(color, image):
        """Return single color channel of image."""
        single_color_image = []

        for row in range(0, len(image) - 1):
            colors = []
            for col in range(0, len(image[row]) - 1):
                colors.append(Embedding.get_color_at(color, image, col, row))
            single_color_image.append(colors)
        return single_color_image

    @staticmethod
    def get_wave_diff(horizontal, vertical):
        """Get wavelet difference from horizontal and vertical."""
        return abs(horizontal - vertical)

    def get_descending_wavelet_diffs(self, horizontals, verticals, num_of_result):
        """Return <number of desired results> wavelet diffs from horizontal and vertical."""
        wavelet_diffs = []
        for i, row in enumerate(horizontals):
            for j, horizontal in enumerate(row):
                wavelet_diffs.append(
                    wavelet_diff.WaveletDiff(
                        Embedding.get_wave_diff(horizontal, verticals[i][j]),
                        j,
                        i
                    )
                )
        descended = sorted(wavelet_diffs, key=lambda x: x.value, reverse=True)
        return descended[0:num_of_result]

    def create_key(self, wavelet_diffs, px_per_row, channel):
        """Create key to get watermark from host on extraction."""
        text = str(channel) + "-"
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
        for row in watermark:
            for pixel in row:
                y = wavelet_diffs[k].y
                x = wavelet_diffs[k].x

                horizontal[y][x], vertical[y][x] = self.embed_0(
                    wavelet_diffs[k].value,
                    threshold0,
                    horizontal[y][x],
                    vertical[y][x],
                    payload0[k]
                ) if pixel == 0 else self.embed_1(
                    wavelet_diffs[k].value,
                    threshold1,
                    horizontal[y][x],
                    vertical[y][x],
                    payload1[k]
                )
                k += 1
        return horizontal, vertical

    def embed_to_subbands(self, watermark, horizontal, vertical, channel):
        """Return created key and watermark-embedded horizontal and vertical."""
        wavelet_diffs = self.get_descending_wavelet_diffs(
            horizontal,
            vertical,
            (len(watermark) ** 2)
        )

        key = self.create_key(wavelet_diffs, len(watermark), channel)

        horizontal, vertical = self.embed_based_on_wavelet_diffs(
            wavelet_diffs,
            horizontal,
            vertical,
            watermark
        )

        return horizontal, vertical, key

    def put_back_color_in_image(self, color, single_color_image, ori_image):
        """Put back watermark embedded color channel to image"""

        for i, row in enumerate(ori_image):
            for j, pixel in enumerate(row):
                pixel[color] = single_color_image[i][j]
        return ori_image

    def save_tiff(self, image, str_key, filename):
        """Save both image and key to tiff file."""
        pil_img = fromarray(uint8(image))
        info = ImageFileDirectory()
        info[self.KEY_LOCATION] = dumps(dict(
            key=str_key
        ))
        pil_img.save(
            filename + ".tif",
            tiffinfo=info
        )

    def ssim(self, host, watermarked):
        """Get similarity between host and watermarked image."""
        mean_host = mean(host)
        mean_watermarked = mean(watermarked)

        std_host = std(host)
        std_watermarked = std(watermarked)
        c_1 = (0.01 * 255) ** 2
        c_2 = (0.03 * 255) ** 2

        return (
            (2 * (mean_host * mean_watermarked) + c_1) * (2 * (std_host * std_watermarked) + c_2)
        ) / (
            (mean_host ** 2 + mean_watermarked ** 2 + c_1) * (std_host ** 2 + std_watermarked ** 2 + c_2)
        )

    @staticmethod
    def normalized_correlation_coef(extracted, watermark):
        """Calculate NC of extracted watermark against the original one."""
        return sum_arr(dot(extracted, watermark) / (norm(extracted) * norm(watermark)))

    def embed_on_particular_channel(self, channel, host, watermark):
        """Function that will be called to embed watermark on particular channel.
        Return watermarked image and ssim."""
        watermarked = deepcopy(host)
        to_be_embedded = Embedding.get_single_color_image(channel, host)

        main, (vertical, horizontal, diagonal) = dwt2(to_be_embedded, 'haar')
        horizontal, vertical, key = self.embed_to_subbands(watermark, horizontal, vertical, channel)
        embedded = idwt2((main, (vertical, horizontal, diagonal)), 'haar')
        # at this point to_be_embedded and embedded are different

        watermarked = self.put_back_color_in_image(channel, embedded, watermarked)

        return (
            watermarked,
            self.ssim(
                host,
                watermarked
            ),
            key
        )

    def embed_watermark(
            self,
            host,
            watermark,
            filename
        ):
        """Function that will return image with best SSIM based on channel."""
        map_result = dict(
            red=self.embed_on_particular_channel(self.RED, host, watermark),
            green=self.embed_on_particular_channel(self.GREEN, host, watermark),
            blue=self.embed_on_particular_channel(self.BLUE, host, watermark)
        )
        result = {}
        max_data = [-1] * 2

        for key, value in map_result.items():
            result[key] = value[1]
            if value[1] > max_data[1]:
                max_data[0] = key
                max_data[1] = value[1]

        # save image that will be used later
        # because openCV flips the order of RGB to BGR
        self.save_tiff(
            cvtColor(
                map_result[max_data[0]][0],
                COLOR_BGR2RGB
            ),
            map_result[max_data[0]][2],
            filename
        )

        result["image"] = imread(filename + ".tif")
        result["max"] = max_data[1]

        return result
