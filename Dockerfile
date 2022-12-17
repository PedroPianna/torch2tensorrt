FROM nvcr.io/nvidia/pytorch:21.11-py3

WORKDIR /home

COPY requirements.txt requirements.txt
RUN python3 -m pip install -r requirements.txt
