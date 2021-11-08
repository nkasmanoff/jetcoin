FROM nvcr.io/nvidia/dli/dli-nano-ai:v2.0.1-r32.6.1


ENV DEBIAN_FRONTEND=noninteractive
ENV SHELL /bin/bash
WORKDIR /home/noah/jetcoin
COPY . /home/noah/jetcoin

# alias python3 -> python
RUN rm /usr/bin/python && \
ln -s /usr/bin/python3 /usr/bin/python && \
ln -s /usr/bin/pip3 /usr/bin/pip
RUN pip install --upgrade pip

RUN pip install -r requirements.txt
