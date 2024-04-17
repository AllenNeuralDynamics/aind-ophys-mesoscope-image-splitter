# hash:sha256:96e9f1462b2bd25613c91599be08d46939f509a85280e4bd26e40ef94fa0b865
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
    pytz==2024.1 \
    scikit-image==0.21.0 \
    scipy==1.10.1 \
    tifffile==2023.7.10
