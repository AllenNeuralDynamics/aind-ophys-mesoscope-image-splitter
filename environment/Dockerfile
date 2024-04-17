# hash:sha256:b90488fcf56abf9d9829b584cf8c87c841825c18b1dbed07b0840bf8400dbf83
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
    numpy \
    pandas \
    pillow==10.2.0 \
    pytz \
    scikit-image \
    scipy \
    tifffile
