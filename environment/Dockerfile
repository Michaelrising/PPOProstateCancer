# hash:sha256:1131e43d711cf87dca8bf0dffc9d85b3ac49c0c8af3c439a05818d5e8757391c
FROM registry.codeocean.com/codeocean/miniconda3:4.9.2-cuda11.7.0-cudnn8-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN conda install -y \
        matplotlib \
        numpy \
        torch==1.10.1 \
    && conda clean -ya

RUN pip install -U --no-cache-dir \
    xitorch
