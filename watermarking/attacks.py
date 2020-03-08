from io import BytesIO
from PIL import Image

class Attacks:

    # PIL
    def compressJPEG(self, img, q):
        buffer = BytesIO()
        img.save(buffer, format="jpeg", quality=q)
        buffer.seek(0)
        return Image.open(BytesIO(buffer.read()))
