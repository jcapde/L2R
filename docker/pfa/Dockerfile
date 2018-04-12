FROM ubuntu:17.10

RUN apt-get update -y \
    && apt-get upgrade -y
#    && apt-get install -y \
    
#&& apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN apt-get install -y \
    cmake \
    libeigen3-dev \
    libboost-all-dev \
    libmatio-dev 

RUN apt-get install -y \
    python3-matplotlib \
    python3-numpy \
    python3-scipy \
    time
    
RUN apt-get install -y \
    python3-dev

RUN apt-get install -y \
    libgsl-dev \
    libblas-dev


RUN mkdir -p /pfa
WORKDIR /pfa
ENV PYTHONPATH /pfa/lib
#ADD ./ /pfa
#RUN /pfa/bin/compile_everything.sh



