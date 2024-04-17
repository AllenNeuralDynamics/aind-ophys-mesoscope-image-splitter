# hash:sha256:206917fc4f59b0062ceb107d3decbcfb0729e322cf4936ef2412eee271318418
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
    pytz==2024.1 \
    scikit-image==0.22.0 \
    scipy==1.12.0 \
    tifffile==2024.2.12
