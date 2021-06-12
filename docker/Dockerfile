FROM nvidia/cuda:11.0-devel-ubuntu20.04

USER root
ENV DEBIAN_FRONTEND=noninteractive
RUN sed -i 's#http://tw.archive.ubuntu.com/#http://archive.ubuntu.com/#' /etc/apt/sources.list && \
    apt update && \
    apt upgrade -y && \
    apt install -y \
        git \
        curl \
        wget \
        vim \
        tmux \
        sudo \
        gosu \
        libgeos-dev \
        libopencv-dev \
        python3 \
        python3-distutils \
    && \
    apt autoremove -y && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

# install pytorch
RUN curl -kL https://bootstrap.pypa.io/get-pip.py | python3 
ENV PATH $PATH:/home/${DOCKER_USER}/.local/bin
RUN pip3 install torch==1.7.0+cu110 torchvision==0.8.1+cu110 torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html

# install DDRNet
WORKDIR /root
RUN curl https://raw.githubusercontent.com/chenjun2hao/DDRNet.pytorch/main/requirements.txt > requirements.txt && \
    sed -i -e 's/opencv-python==3.4.1.15/opencv-python==3.4.8.29/g' ./requirements.txt && \
    pip3 install -r requirements.txt && \
    rm requirements.txt
RUN ln -s /usr/bin/python3 /usr/bin/python 

# install ROS noetic
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu focal main" > /etc/apt/sources.list.d/ros-latest.list' && \
    apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654 && \
    apt update && \
    apt install -y \
        ros-noetic-desktop-full \
        python3-argcomplete \
    && \
    apt autoremove -y && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

RUN echo source /opt/ros/noetic/setup.bash >> /root/.bashrc
