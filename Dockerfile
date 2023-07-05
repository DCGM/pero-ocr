FROM debian:bullseye

WORKDIR /pero-ocr/
COPY ./ /pero-ocr/

RUN apt-get update && \
    apt-get install -y --no-install-recommends python3-pip ffmpeg && \
    apt-get clean

RUN pip install --no-cache --upgrade pip
RUN pip wheel --no-cache -w arch .
RUN pip install arch/*
RUN rm -rf arch
RUN rm -rf /root/.cache

CMD ["/usr/bin/python3", "user_scripts/parse_folder.py", "--help"]
