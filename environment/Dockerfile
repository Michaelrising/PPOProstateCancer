# hash:sha256:f23f8f1c8e0a5f18fbcfd59aa5172b0f84ef9e448501f1334a6269b430c1aafd
FROM registry.codeocean.com/codeocean/miniconda3:4.9.2-cuda11.7.0-cudnn8-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN conda install -y \
        argparse \
        collections \
        datetime \
        gym \
        json \
        matplotlib \
        numpy \
        os \
        pandas \
        scipy \
        seaborn \
        torch==1.10.1 \
    && conda clean -ya

RUN pip install -U --no-cache-dir \
    xitorch
