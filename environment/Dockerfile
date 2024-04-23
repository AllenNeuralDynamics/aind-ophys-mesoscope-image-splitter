# hash:sha256:4b17746a6f5fb619d7a76e580a6b4e60a8a02f93f6900df399c368936ad626ae
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.0.9-miniconda4.9.2-python3.8-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        vim=2:8.1.2269-1ubuntu5.22 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -U --no-cache-dir \
    aind-data-schema==0.29.0 \
    aind-ophys-utils==0.0.7 \
    h5py==3.10.0 \
    numpy==1.24.4 \
    pandas==2.0.3 \
    pillow==10.2.0 \
    pydantic==2.7.1 \
    pydantic-settings==2.2.1 \
    pytz==2024.1 \
    scikit-image==0.21.0 \
    scipy==1.10.1 \
    tifffile==2024.2.12
