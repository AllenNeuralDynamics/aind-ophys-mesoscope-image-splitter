# hash:sha256:dc1d850afbc4ea7b144d8a2d7b7b5679d21fc01254178a070aae98f3e29520a0
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/py-julia-r:python3.10.12-R4.2.3-julia1.7.0-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        vim=2:8.2.3995-1ubuntu2.16 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -U --no-cache-dir \
    aind-ophys-utils==0.0.7 \
    h5py==3.11.0 \
    pydantic==2.7.1 \
    pydantic-settings==2.2.1 \
    scikit-image==0.22.0 \
    tifffile==2024.2.12
