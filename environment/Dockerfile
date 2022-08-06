# hash:sha256:e3381e6d46f0ef57f887514d50222a42b25c1935bac7d484f629641ff9b2d7b0
FROM registry.codeocean.com/codeocean/miniconda3:4.9.2-cuda11.7.0-cudnn8-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN conda install -y \
        matplotlib==3.5.1 \
        numpy==1.21.5 \
        pandas==1.4.2 \
        scipy==1.7.3 \
        seaborn==0.11.2 \
    && conda clean -ya

RUN pip install -U --no-cache-dir \
    gym==0.21.0 \
    tensorboard \
    xitorch==0.3.0

COPY postInstall /
RUN /postInstall
