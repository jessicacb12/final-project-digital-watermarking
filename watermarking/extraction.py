"""This script is used to extract watermark from watermarked image."""

from re import findall, search
from math import sqrt
from pywt import dwt2
from PIL.Image import open as open_pil
from PIL.Image import fromarray
from numpy import uint8, array
from watermarking import embedding, cnn, forward, process

class Extraction:
    """Contains CNN and the rest of watermark extraction methods."""

    FILENAME = "data/watermarked_extracted"

    def get_positions_from_key(self, key):
        """Get positions from text key."""
        combined_positions = findall('([0-9]+,[0-9]+)', key)
        if len(combined_positions) == 0:
            return None

        positions = []
        for combined in combined_positions:
            str_positions = combined.split(",")
            positions.append(
                [int(str_positions[0]), int(str_positions[1])]
            )
        return positions

    def extract_channel_from_key(self, key):
        """Extract channel from extracted key."""
        return search(
            '[0-3]$',
            search('{"key": "[0-3]', key).group()
        ).group()

    @staticmethod
    def extract_key_from_image_file(location):
        """Extract key from KEY_LOCATION tag from image file"""
        img = open_pil(location)
        return img.tag_v2[embedding.Embedding.KEY_LOCATION]

    @staticmethod
    def extract_key_from_image_description(img):
        """Extract key from ImageDescription tag in TIFF watermarked image."""
        key = findall('{"key":.*}', img)
        return key[0] if len(key) > 0 else None

    def extract_embedding_map(self, watermarked, key):
        """Extract embedding map to be used as CNN input later."""
        _, (vertical, horizontal, __) = dwt2(watermarked, 'haar')
        embedding_map = []

        side = int(sqrt(len(key)))
        i = 0
        row = []

        # make into 2 dimensions with assumption that watermark is square matrix
        for position in key:
            value = embedding.Embedding.get_wave_diff(
                horizontal[position[1]][position[0]],
                vertical[position[1]][position[0]]
            )
            row.append(
                value
            )
            if i == side - 1:
                i = 0
                embedding_map.append(row)
                row = []
            else:
                i += 1
        return embedding_map

    def save_tiff(self, image):
        """Save only image to tiff file."""
        pil_img = fromarray(uint8(image))
        pil_img.save(
            process.Process.ROOT + self.FILENAME + ".tif"
        )

    def extract_watermark(self, watermarked, key):
        """Extract watermark from watermarked image."""
        positions = self.get_positions_from_key(key)
        channel = None
        try:
            channel = self.extract_channel_from_key(key)
        except AttributeError:
            return "Channel is not detected in key"

        if positions is None:
            return "Locations are not detected in key"

        embedding_map = self.extract_embedding_map(
            embedding.Embedding.get_single_color_image(
                int(channel),
                watermarked
            ),
            self.get_positions_from_key(key)
        )
        extracted = forward.Forward(
            False,
            [[embedding_map]], # double array as batch and channel
            cnn.CNN.init_params()
        ).run()
        print('shape: ', array(extracted).shape)
        self.save_tiff(extracted)
        return extracted
