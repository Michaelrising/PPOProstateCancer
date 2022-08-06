# hash:sha256:42b0124ac4485a9c2e821ef30bb4b58f69280175a9878b85872b70ddcf6555a7
FROM registry.codeocean.com/codeocean/miniconda3:4.9.2-cuda11.7.0-cudnn8-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN conda install -y \
        numpy \
        torch==1.10.1 \
    && conda clean -ya
