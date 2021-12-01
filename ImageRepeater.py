import ImageProvider
from PIL import Image
import io
import numpy as np


class ImageRepeater(ImageProvider.ImageProvider):
    def get_next_image(self) -> bytes:
        image_bytes = open('Tests/Images/img1.jpg', "rb").read()
        img = Image.open(io.BytesIO(image_bytes))
        np_image = np.asarray(img)
        return np_image

    def stop(self):
        pass


