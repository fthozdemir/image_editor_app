from io import BytesIO
from PIL import Image
from imageio import imread
import base64


def base64_to_image(b64String):
    img = imread(BytesIO(base64.b64decode(b64String)))

    return img


def image_to_base64(img):
    pil_img = Image.fromarray(img)
    buff = BytesIO()
    pil_img.save(buff, format="PNG")
    b64string = base64.b64encode(buff.getvalue()).decode("utf-8")

    return b64string
