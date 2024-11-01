FROM python:3.12

WORKDIR /app

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install git+https://github.com/openai/CLIP.git
RUN pip install git+https://github.com/CASIA-IVA-Lab/FastSAM.git
RUN git clone https://github.com/ultralytics/ultralytics.git
RUN cd ultralytics && pip install -e . && cd ..
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install ultralytics==8.0.100

COPY . .

EXPOSE 5000

CMD ["python", "app.py"]
