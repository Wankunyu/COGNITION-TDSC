# encoding: utf-8

import base64
from pathlib import Path
from PIL import Image
from typing import List, Dict, Any, Optional, Union, Callable


def encode_image_to_base64(image_path: str, head: str = "") -> str:
    """
    Encode an image to base64 string.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        str: Base64 encoded string of the image
        
    Raises:
        FileNotFoundError: If the image file doesn't exist
        Exception: If there's an error reading the file or encoding
    """
    try:
        # Convert string path to Path object
        path = Path(image_path)
        
        # Check if file exists
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Read the image file in binary mode
        with open(path, 'rb') as image_file:
            # Encode the binary data to base64
            encoded_string = base64.b64encode(image_file.read())
            
            # Convert bytes to string
            return head + encoded_string.decode('utf-8')
            
    except FileNotFoundError as e:
        raise e
    except Exception as e:
        raise Exception(f"Error encoding image: {str(e)}")

def add_to_image(
    image_path: str, 
    box_position: List[List[int]], 
    add_image_path: str, 
    output_image_path: str,
    opacity:float=0.7
    ) -> None:
    """
    Add a tick image to a specified box position in the main image
    
    Parameters:
    image_path: str - path to the main image
    box_position: list - [(left, top, right, bottom), ...] coordinates of the boxes
    add_image_path: str - path to the image to add
    opacity: float - opacity level for the tick (0.0 to 1.0)
    
    Returns:
    PIL.Image - The resulting image with the tick added
    """
    try:
        main_image = Image.open(image_path).convert('RGBA')
        
        add_image = Image.open(add_image_path).convert('RGBA')

        result = main_image
        
        for pos in box_position:
            add_image = Image.open(add_image_path).convert('RGBA')
            # Get box dimensions
            left, top, right, bottom = pos
            box_width = right - left
            box_height = bottom - top
            
            # Resize tick image to fit the box
            add_image = add_image.resize((box_width, box_height), Image.Resampling.LANCZOS)
            
            # Create a new transparent layer for the tick
            add_layer = Image.new('RGBA', main_image.size, (0, 0, 0, 0))
            
            # Adjust opacity of tick image
            if opacity < 1.0:
                add_data = add_image.getdata()
                new_data = []
                for item in add_data:
                    # Preserve the RGB values but adjust the alpha channel
                    new_data.append((item[0], item[1], item[2], int(item[3] * opacity)))
                add_image.putdata(new_data)
            
            # Paste the tick image onto the transparent layer
            add_layer.paste(add_image, (left, top))
            
            # Combine the main image with the tick layer
            result = Image.alpha_composite(result, add_layer)
        
        result.save(output_image_path)
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return None


def combine_number_images(left_image_path, right_image_path):
    # 打开两个图片
    img1 = Image.open(left_image_path).convert('RGBA')
    img2 = Image.open(right_image_path).convert('RGBA')

    # 获取图片尺寸
    width1, height1 = img1.size
    width2, height2 = img2.size

    # 创建新图片，宽度是两个图片宽度之和，高度取最大值
    new_width = width1 + width2
    new_height = max(height1, height2)
    new_img = Image.new('RGBA', (new_width, new_height), 'white')

    # 将两个图片粘贴到新图片上
    new_img.paste(img1, (0, 0))
    new_img.paste(img2, (width1, 0))

    return new_img
