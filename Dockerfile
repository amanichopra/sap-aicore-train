FROM python:3.12-slim

ENV HOME=/root

WORKDIR /app/src/
COPY train.py requirements.txt ./
RUN pip3 install -r requirements.txt

WORKDIR $HOME

