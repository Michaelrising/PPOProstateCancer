# hash:sha256:7fb7a335a2e8a6cbf3ffc80d7b7a7385572f78ea458a85b06942dc2e3b757bcf
FROM registry.codeocean.com/codeocean/miniconda3:4.9.2-cuda11.7.0-cudnn8-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN conda install -y \
        argparse \
        datetime \
        gym \
        matplotlib \
        numpy \
        os \
        pandas \
        torch==1.10.1 \
    && conda clean -ya

RUN pip install -U --no-cache-dir \
    xitorch
