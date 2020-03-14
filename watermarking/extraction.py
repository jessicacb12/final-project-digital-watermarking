"""This script is used to extract watermark from watermarked image."""

from re import findall
from math import sqrt
from pywt import dwt2
from watermarking import embedding

class Extraction:
    """Contains CNN and the rest of watermark extraction methods."""

    def get_positions_from_key(self, key):
        """Get positions from text key."""
        combined_positions = findall('([0-9]+,[0-9]+)', key)
        positions = []
        for combined in combined_positions:
            str_positions = combined.split(",")
            positions.append(
                [int(str_positions[0]), int(str_positions[1])]
            )
        return positions

    @staticmethod
    def extract_key_from_image_description(img):
        """Extract key from ImageDescription tag in TIFF watermarked image."""
        key = findall('{"key":.*}', img)
        return key[0] if len(key) > 0 else None

    def extract_embedding_map(self, watermarked, key):
        """Extract embedding map to be used as CNN input later."""
        main, (vertical, horizontal, diagonal) = dwt2(watermarked, 'haar')
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

    def extract_watermark(self, watermarked, key):
        """Extract watermark from watermarked image."""

        #hardcoded color channel. need to be fixed later
        embedding_map = self.extract_embedding_map(
            embedding.Embedding.get_single_color_image(
                embedding.Embedding.GREEN,
                watermarked
            ),
            self.get_positions_from_key(key)
        )
