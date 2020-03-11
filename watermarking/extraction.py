"""This script is used to extract watermark from watermarked image."""

from re import findall

class Extraction:
    """Contains CNN and the rest of watermark extraction methods."""

    def get_positions_from_key(self, key):
        """Get positions from text key."""
        combined_positions = findall("(\w,\w)", key)
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

    def extract_watermark(self, watermarked):
        """Extract watermark from watermarked image."""
        print(watermarked, flush=True)
