# hash:sha256:72272501e38e57b130bd92eb6fd4c2c261cc8ca3d37474017ebda04bb97224a0
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
