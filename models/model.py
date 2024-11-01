from typing import Any, List
from fastsam import FastSAM
from PIL import Image
from fastsam import FastSAM, FastSAMPrompt
import numpy as np
from configs.config import DEVICE, FAST_SAM_CONF, FAST_SAM_IMGSZ, FAST_SAM_IOU, FAST_SAM_RETINA_MASKS, MODEL_PATH
from PIL import Image
import os


def loadModel(model_name: str = "FastSAM-x.pt") -> Any:
    path: str = MODEL_PATH + "/" + model_name
    if not os.path.exists(path):
        downloadModel(path)
    return FastSAM(f"{MODEL_PATH}/{model_name}")


def getMask(image_path: Image, fast_sam: FastSAM, point: List[List[int]], point_label: List[int]) -> Any:
    result: Any = fast_sam(
        source=image_path,
        device=DEVICE,
        retina_masks=FAST_SAM_RETINA_MASKS,
        imgsz=FAST_SAM_IMGSZ,
        conf=FAST_SAM_CONF,
        iou=FAST_SAM_IOU,
    )
    prompt_process = FastSAMPrompt(image_path, result, device=DEVICE)
    return prompt_process.point_prompt(points=point, pointlabel=point_label)


def downloadModel(model_name):
    import requests
    url = "https://firebasestorage.googleapis.com/v0/b/lexicons-5.appspot.com/o/FastSam-Models%2FFastSAM-x.pt?alt=media&token=64b65560-17d6-47b0-8a2b-8e2ee096da64"
    r = requests.get(url)
    with open(model_name, 'wb') as f:
        f.write(r.content)
    return model_name
