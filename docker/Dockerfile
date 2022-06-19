FROM nvidia/cuda:11.2.0-devel-ubuntu20.04
# FROM --platform=linux/amd64 ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

# Force color prompt in terminal.
RUN sed -i 's/#force_color_prompt=yes/force_color_prompt=yes/' /root/.bashrc

# japanise setting
RUN apt-get update && apt-get install -y locales
RUN localedef -f UTF-8 -i ja_JP ja_JP.UTF-8
ENV LANG ja_JP.UTF-8
ENV LANGUAGE ja_JP:ja
ENV LC_ALL ja_JP.UTF-8
ENV TZ JST-9
ENV TERM xterm

# install utility
RUN apt-get update && \
    apt-get install -y \
    wget \
    curl \
    git \
    vim \
    python3-pip \
    screen

RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC

# fix opencv error
RUN apt-get update && \
    apt-get install -y \
    libgl1-mesa-dev \
    libglib2.0-0
# RUN apt-get update && apt-get upgrade -y
# RUN apt-get install -y libgl1-mesa-dev
# RUN apt-get update && apt-get install -y libglib2.0-0

# Add the following line to get native library of OpenCV.
# RUN apt-get install -y libopencv-dev

# install fish
RUN apt-get install -y fish
SHELL ["/usr/bin/fish", "-c"]

# install fish plagin
RUN curl https://git.io/fisher --create-dirs -sLo ~/.config/fish/functions/fisher.fish
RUN fisher install oh-my-fish/theme-bobthefish
RUN fisher install jethrokuan/z

# install python package
WORKDIR /root/requirements
ADD requirements.txt /root/requirements
RUN pip3 install -r requirements.txt
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

WORKDIR /root/workspace
