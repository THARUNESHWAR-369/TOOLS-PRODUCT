import torch
import os

DEVICE = torch.device("cpu")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))

MODEL_PATH = os.path.join(BACKEND_DIR, 'models', 'checkpoint')

FAST_SAM_IMGSZ = 1024
FAST_SAM_CONF = 0.5
FAST_SAM_IOU = 0.6
FAST_SAM_RETINA_MASKS = True
