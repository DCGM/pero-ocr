FROM debian:bullseye

WORKDIR /opt/pero/pero-ocr/
COPY ./ /opt/pero/pero-ocr/

RUN cat /etc/apt/sources.list | sed -e 's/main/main contrib non-free/g' > /etc/apt/tmp.list && mv /etc/apt/tmp.list /etc/apt/sources.list

RUN apt-get update && \
    apt-get install -y --no-install-recommends python3-pip ffmpeg && \
    apt-get clean

RUN pip install --no-cache --upgrade pip && \
    pip wheel --no-cache -w arch . && \
    pip install arch/* && \
    rm -rf arch && \
    rm -rf /root/.cache

CMD ["/usr/bin/python3", "user_scripts/parse_folder.py", "--help"]
