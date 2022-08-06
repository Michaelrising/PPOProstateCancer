# hash:sha256:0b6bb816938891e68c6f14821c0546d0a33bb1b4435c3e561d6f5d6f8f0d403d
FROM registry.codeocean.com/codeocean/miniconda3:4.9.2-cuda11.7.0-cudnn8-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN conda install -y \
        matplotlib \
        numpy \
        pandas \
        scipy \
        seaborn \
    && conda clean -ya

RUN pip install -U --no-cache-dir \
    gym==0.21.0 \
    xitorch

COPY postInstall /
RUN /postInstall
