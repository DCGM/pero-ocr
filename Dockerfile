FROM debian:bullseye

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /opt/pero/pero-ocr/
COPY ./ /opt/pero/pero-ocr/

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip python3-dev build-essential ffmpeg \
  && python3 -m pip install --upgrade pip setuptools wheel \
  && pip install --no-cache-dir --prefer-binary . \
  && apt-get purge -y --auto-remove python3-dev build-essential \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/* /root/.cache

CMD ["/usr/bin/python3", "user_scripts/parse_folder.py", "--help"]
