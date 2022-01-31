# Official jupyter image that contains common packages for data analysis
FROM jupyter/datascience-notebook:notebook-6.4.7

# root user to enable installation of packages
USER root
RUN apt-get update && apt-get -y install python3-pip ffmpeg
WORKDIR /app

# Install python packages
COPY requirements/ requirements/
RUN pip3 install --no-cache-dir -r requirements/requirements.txt
RUN pip3 install --no-cache-dir -r requirements/requirements-dev.txt

