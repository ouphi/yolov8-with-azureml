FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest

ENV DEBIAN_FRONTEND noninteractive
RUN apt update
RUN TZ=Etc/UTC apt install -y tzdata
RUN apt install --no-install-recommends -y gcc git zip curl htop libgl1-mesa-glx libglib2.0-0 libpython3-dev gnupg g++
RUN apt upgrade --no-install-recommends -y openssl tar
RUN pip install azureml-inference-server-http
RUN pip install ultralytics==8.0.158
