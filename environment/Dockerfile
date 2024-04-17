# hash:sha256:753fb97d937b93776439b1b0577f976767f343f68fea487898469d38a5823b9e
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
    pandas==2.2.0 \
    pillow==10.2.0 \
    python-dateutil==2.8.2 \
    pytz==2024.1 \
    scikit-image==0.22.0 \
    scipy==1.12.0 \
    tifffile==2024.2.12
