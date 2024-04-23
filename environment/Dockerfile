# hash:sha256:5cc1ac9dd0fd34bc684fd8067eb7d0e2245cac5144b85d98bb17b29ad5a4e6f4
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/py-julia-r:python3.10.12-R4.2.3-julia1.7.0-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        apt=2.0.5 \
        base-passwd=3.5.47 \
        bash=5.0-6ubuntu1.1 \
        bsdutils=1:2.34-0.1ubuntu9.1 \
        build-essential=12.8ubuntu1.1 \
        ca-certificates=20210119~20.04.1 \
        coreutils=8.30-3ubuntu2 \
        curl=7.68.0-1ubuntu2.5 \
        dash=0.5.10.2-6 \
        debconf=1.5.73 \
        diffutils=1:3.7-3 \
        e2fsprogs=1.45.5-2ubuntu1 \
        fdisk=2.34-0.1ubuntu9.1 \
        findutils=4.7.0-1ubuntu1 \
        git=1:2.25.1-1ubuntu3.1 \
        gnupg2=2.2.19-3ubuntu2.1 \
        grep=3.4-1 \
        gzip=1.10-0ubuntu4 \
        hostname=3.23 \
        less=551-1ubuntu0.1 \
        libattr1=1:2.4.48-5 \
        libext2fs2=1.45.5-2ubuntu1 \
        libpam-modules-bin=1.3.1-5ubuntu4.1 \
        libpam-runtime=1.3.1-5ubuntu4.1 \
        libpcre3=2:8.39-12build1 \
        libss2=1.45.5-2ubuntu1 \
        locales=2.31-0ubuntu9.2 \
        mawk=1.3.4.20200120-2 \
        mount=2.34-0.1ubuntu9.1 \
        nano=4.8-1ubuntu1 \
        ncurses-base=6.2-0ubuntu2 \
        ncurses-bin=6.2-0ubuntu2 \
        procps=2:3.3.16-1ubuntu2.1 \
        sed=4.7-1 \
        sensible-utils=0.0.12+nmu1 \
        sysvinit-utils=2.96-2.1ubuntu1 \
        unzip=6.0-25ubuntu1 \
        vim=2:8.1.2269-1ubuntu5.22 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -U --no-cache-dir \
    aind-ophys-utils==0.0.7 \
    h5py==3.11.0 \
    pydantic==2.7.1 \
    pydantic-settings==2.2.1 \
    scikit-image==0.22.0 \
    tifffile==2024.2.12
