FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ARG VENV_NAME="cosyvoice"
ENV VENV=$VENV_NAME
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

ENV DEBIAN_FRONTEN=noninteractive
ENV PYTHONUNBUFFERED=1
SHELL ["/bin/bash", "--login", "-c"]

RUN apt-get update -y --fix-missing
RUN apt-get install -y build-essential ffmpeg sox libsox-dev python3.10 python3-pip && \
    apt-get clean

WORKDIR /workspace

ENV PYTHONPATH="${PYTHONPATH}:/workspace/CosyVoice:/workspace/CosyVoice/third_party/Matcha-TTS"

COPY . /workspace/CosyVoice

RUN cd CosyVoice && \
    pip install -r requirements.txt
RUN sh /workspace/CosyVoice/scripts/setup.sh
WORKDIR /workspace/CosyVoice
