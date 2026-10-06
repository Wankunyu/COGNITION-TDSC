
from captcha.utils.image_utils import encode_image_to_base64
from pathlib import Path

def is_in_box(box, point):
    x1, y1, x2, y2 = box
    return x1 <= point[0] <= x2 and y1 <= point[1] <= y2

def gen_image_message(image_list):
    result = []
    for image in image_list:
        result.append(
            {
                "type": "image_url",
                "image_url": {
                    # "url": encode_image_to_base64(image, head="data:image/base64;base64,")
                    "url": encode_image_to_base64(image, head="data:image;base64,") #qwen格式
                    # "url": encode_image_to_base64(image, head="data:image/png;base64,")

                }
            }
        )

    return result

