FROM debian:bullseye

WORKDIR /opt/pero/pero-ocr/
COPY ./ /opt/pero/pero-ocr/

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3-pip python3-dev \
        build-essential g++ \
        cmake ninja-build pkg-config \
        libopenblas-dev \
        ffmpeg && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip && \
    pip wheel --no-cache-dir --prefer-binary -w arch . && \
    pip install --no-cache-dir arch/* && \
    rm -rf arch /root/.cache

CMD ["/usr/bin/python3", "user_scripts/parse_folder.py", "--help"]
