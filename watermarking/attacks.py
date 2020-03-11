"""This script is used to digitally transform watermarked image."""

from io import BytesIO
from PIL import Image

class Attacks:
    """Contains attacks for watermarked image."""

    def compress_jpeg(self, img, quality):
        """Return PIL JPEG-compressed image."""
        buffer = BytesIO()
        img.save(buffer, format="jpeg", quality=quality)
        buffer.seek(0)
        return Image.open(BytesIO(buffer.read()))
