
import base64
import io
from typing import Any

def convertToBuffer(image : Any) -> bytes:
    buffered : io.BytesIO = io.BytesIO()
    image.save(buffered, format="PNG")
    img_base64 : bytes = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_base64