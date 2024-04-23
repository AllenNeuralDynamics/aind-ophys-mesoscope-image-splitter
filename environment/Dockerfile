# hash:sha256:98b6e64c933a2c0018e09e81f4d9c15f513efc22af79a8b814dd801ddb267c25
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.0.9-miniconda4.9.2-python3.8-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    aind-data-schema==0.29.0 \
    aind-ophys-utils==0.0.7 \
    h5py==3.10.0 \
    numpy==1.24.4 \
    pandas==2.0.3 \
    pillow==10.2.0 \
    pydantic \
    pydantic-settings \
    pytz==2024.1 \
    scikit-image==0.21.0 \
    scipy==1.10.1 \
    tifffile==2023.7.10
