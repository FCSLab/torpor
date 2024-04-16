# Copyright 2018 gRPC authors.
# Copyright 2018 Claudiu Nedelcu.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.#
#
# Based on https://hub.docker.com/r/grpc/cxx.

FROM nvcr.io/nvidia/pytorch:22.01-py3

USER root

RUN apt-get update && apt-get install -y \
  autoconf \
  automake \
  build-essential \
  # cmake \
  curl \
  g++ \
  git \
  libtool \
  make \
  pkg-config \
  unzip \
  g++  \
  libgflags-dev \
  libgtest-dev  \
  libzmq3-dev \
  clang  \
  libc++-dev \
  && apt-get clean

RUN wget https://github.com/Kitware/CMake/releases/download/v3.24.1/cmake-3.24.1-linux-x86_64.tar.gz \
    && tar -zxvf cmake-3.24.1-linux-x86_64.tar.gz \
    && ln -sf /cmake-3.24.1-linux-x86_64/bin/* /usr/bin \
    && cmake --version


RUN git clone --recurse-submodules -b v1.46.3 --depth 1 --shallow-submodules https://github.com/grpc/grpc /var/local/git/grpc && \
    cd /var/local/git/grpc && \
    git submodule update --init --recursive

RUN echo "-- installing protobuf" && \
    cd /var/local/git/grpc/third_party/protobuf && \
    ./autogen.sh && ./configure --enable-shared && \
    make -j$(nproc) && make -j$(nproc) check && make install && make clean && ldconfig

RUN echo "-- installing grpc" && \
    cd /var/local/git/grpc && \
    mkdir build && \
    cd build && \
    cmake -DCMAKE_INSTALL_PREFIX=/usr/local .. && \
    make -j$(nproc) && make install && make clean && ldconfig

ENV PROJ_HOME /gpu-swap
RUN mkdir -p $PROJ_HOME

COPY . $PROJ_HOME
WORKDIR $PROJ_HOME

