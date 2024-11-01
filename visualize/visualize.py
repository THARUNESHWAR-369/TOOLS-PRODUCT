from PIL import Image
import cv2
import numpy as np
import torch
from typing import Any, Dict, List

def removeBgFromSegmentImage(og_image : Any, og_mask : Any, color: tuple = (0, 0, 255, 255),  opacity: float = 0.2) -> Image:
    og_image = np.array(og_image.convert('RGB'))
    mask = og_mask.astype(np.uint8) * 255  # Convert to 0 or 255

    rgba_image = np.zeros((og_image.shape[0], og_image.shape[1], 4), dtype=np.uint8)

    color_with_opacity = (color[0], color[1], color[2], int(color[3] * opacity))
    rgba_image[mask > 0] = color_with_opacity

    return Image.fromarray(rgba_image)

def removeOnlyBg(og_image : Any, og_mask : Any) -> Image:
    img = np.array(og_image.convert('RGB'))
    mask = cv2.resize(og_mask.astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    rgba_image = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.uint8)
    rgba_image[..., :3] = img
    rgba_image[..., 3] = mask * 255

    return Image.fromarray(rgba_image)
