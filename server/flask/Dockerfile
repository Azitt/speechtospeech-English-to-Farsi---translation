# Pull tensorflow image with Python3
FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime



# Install audio libraries
#RUN apt-get update && apt-get install -y libsndfile1 libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0 ffmpeg
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get -y install cmake protobuf-compiler gcc mono-mcs python3-dev libcups2-dev gcc \
    libpangocairo-1.0-0 gcc build-essential libpython3-dev libdbus-1-dev \
    libglib2.0-dev libcairo2-dev libgirepository1.0-dev libsndfile1-dev libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0 ffmpeg

COPY requirements2.txt /
RUN pip install -r /requirements2.txt
# Transfer content from current dir to /app in container
ADD . /app

# Set the working directory to /app
WORKDIR /app
# Install python packages


#RUN pip install -q espnet==0.10.3 pyopenjtalk==0.1.5 parallel_wavegan==0.5.3 espnet_model_zoo
ENV FLASK_APP=Server
ENTRYPOINT ["flask","run","--host=0.0.0.0","--port=5000"]




