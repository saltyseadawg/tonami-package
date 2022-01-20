# slim so smaller pls
FROM python:3.9-slim-buster

RUN apt-get update && apt-get -y install python3-pip libsndfile1 ffmpeg
WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

CMD ["python3"]
