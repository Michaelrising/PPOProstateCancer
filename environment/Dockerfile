# hash:sha256:b2eb4b29d280f4f4ed027717fbcf1e9b74b0122bd96f462d986d141ae04481d1
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
    && conda clean -ya

RUN pip install -U --no-cache-dir \
    xitorch

COPY postInstall /
RUN /postInstall
