from typing import Any, Dict, List
from fastsam import FastSAM, FastSAMPrompt
import numpy as np
from configs.config import DEVICE, FAST_SAM_CONF, FAST_SAM_IMGSZ, FAST_SAM_IOU, FAST_SAM_RETINA_MASKS

def preprocess(points_data: List[Dict]) -> Any:
    
    input_points = []
    input_labels = []
    
    for point in points_data:
        input_points.append([int(point['x_']), int(point['y_'])])
        input_labels.append(int(point['flag_']))
    
    return input_points, input_labels

    
