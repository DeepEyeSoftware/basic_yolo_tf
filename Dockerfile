FROM tensorflow/tensorflow:latest-gpu-jupyter

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

COPY clean_yolov4/requirements-gpu.txt .
RUN pip3 install -r requirements-gpu.txt

RUN mkdir -p /opt/ml/input/data/train
RUN mkdir -p /opt/ml/input/data/eval
RUN mkdir -p /opt/ml/model
RUN mkdir -p /opt/ml/output

COPY /clean_yolov4/core /opt/ml/code/core
COPY /clean_yolov4/create_estimator.py /opt/ml/code

ENV PATH="/opt/ml/code:${PATH}"
ENV SM_MODEL_DIR="/opt/ml/model"
ENV SM_CHANNEL_TRAIN="/opt/ml/input/data/train/"
ENV SM_CHANNEL_EVAL="/opt/ml/input/data/eval"
ENV SM_CHANNEL_COMMON="/opt/ml/input/data/common/"

WORKDIR '/opt/ml/code'

ENTRYPOINT ["python", "create_estimator.py"]