# slim so smaller pls
FROM jupyter/datascience-notebook:notebook-6.4.7

USER root
RUN apt-get update && apt-get -y install python3-pip ffmpeg
WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

