# hash:sha256:2c2421217efa6dd6428185dfc377c1d9fd35195cc04831a5a6a04acbeb4ccf8f
FROM registry.codeocean.com/codeocean/miniconda3:4.9.2-cuda11.7.0-cudnn8-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN conda install -y \
        jupyterlab==3.4.4 \
        matplotlib==3.5.1 \
        numpy==1.21.5 \
        pandas==1.4.2 \
        scipy==1.7.3 \
        seaborn==0.11.2 \
    && conda clean -ya

RUN pip install -U --no-cache-dir \
    gym==0.21.0 \
    plotly \
    sklearn \
    tensorboard==2.9.1 \
    xitorch==0.3.0

COPY postInstall /
RUN /postInstall
