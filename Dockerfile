# Build container:
# $ docker build -t reidtracker .

# Run it (nvidia runtime is required):
# $ docker run --runtime=nvidia -it --rm --name reid -e DISPLAY=:0 -e QT_X11_NO_MITSHM=1 -v /tmp/.X11-unix:/tmp/.X11-unix -i reidtracker:latest /bin/bash

FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

RUN apt update
RUN apt install -y --no-install-recommends \
    git vim tar unzip wget \
    ffmpeg \
    python3-dev python3-pip \
    libsm6 libxext6 libxrender-dev

RUN pip3 install --upgrade pip
RUN pip3 install setuptools
RUN pip3 install torch torchvision
RUN pip3 install opencv-python==4.2.0.32 ffmpeg-python onnx psutil
RUN pip3 install Cython h5py Pillow six scipy tb-nightly yacs gdown flake8 yapf isort && \
    git clone https://github.com/KaiyangZhou/deep-person-reid.git && \
    cd deep-person-reid && \
    python3 setup.py develop
RUN pip3 install faiss-cpu


ADD ./models/ /opt/models/
#RUN mkdir -p /root/.cache/torch/checkpoints/ && \
#    cp /opt/models/fpn_osnet_x1_0_imagenet.pth /root/.cache/torch/checkpoints/

    
