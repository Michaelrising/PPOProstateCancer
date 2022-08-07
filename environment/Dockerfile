# hash:sha256:d42d084cddb08d1ae62675a66014220a0b54145dba3d1eb18a99777d2bef97db
FROM registry.codeocean.com/codeocean/miniconda3:4.9.2-cuda11.7.0-cudnn8-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        cm-super=0.3.4-15 \
        dvipng=1.15-1.1 \
        texlive-fonts-recommended=2019.20200218-1 \
        texlive-latex-extra=2019.202000218-1 \
    && rm -rf /var/lib/apt/lists/*

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
    plotly==5.9.0 \
    scienceplots==1.0.9 \
    sklearn==0.0 \
    tensorboard==2.9.1 \
    xitorch==0.3.0

COPY postInstall /
RUN /postInstall
