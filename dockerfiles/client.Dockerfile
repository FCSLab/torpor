FROM standalone-base:latest

RUN pip3 install torch==1.11.0a0+bfe5ad2
RUN pip3 install transformers==4.27.1
RUN pip3 install pytorch-pretrained-bert==0.6.2

RUN rm -rf $PROJ_HOME
COPY . $PROJ_HOME
WORKDIR $PROJ_HOME

RUN cd ${PROJ_HOME} && \
    rm -rf build && \
    mkdir build && \
    cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_TARGET:STRING=client .. && \
    make

RUN rm -rf /client_bin && \
    mkdir /client_bin && \
    cp ${PROJ_HOME}/build/src/client/librtclient.so /client_bin && \
    cd /client_bin && \
    ldd librtclient.so | grep "=> /" | awk '{print $3}' | xargs -I '{}' cp -v '{}' . && \
    cp ${PROJ_HOME}/build/src/client/libmycuda.so /usr/lib/x86_64-linux-gnu/libcuda.so.1 && \
    echo "LD_LIBRARY_PATH=/client_bin/:\$LD_LIBRARY_PATH LD_PRELOAD=/client_bin/librtclient.so python \$1" > start.sh && \
    echo "LD_LIBRARY_PATH=/client_bin/:\$LD_LIBRARY_PATH LD_PRELOAD=/client_bin/librtclient.so CLIENT_ID=\$1 python \$2 \$3" > start_with_id.sh && \
    echo "LD_LIBRARY_PATH=/client_bin/:\$LD_LIBRARY_PATH LD_PRELOAD=/client_bin/librtclient.so CLIENT_ID=\$1 CUR_SERVER_ID=\$2 python \$3 \$4" > start_with_server_id.sh && \
    chmod +x start.sh && \
    chmod +x start_with_id.sh && \
    chmod +x start_with_server_id.sh && \
    mv /client_bin/start.sh /client_bin/start_with_id.sh /client_bin/start_with_server_id.sh /

RUN cp ${PROJ_HOME}/tools/kernel_info.txt /
RUN cp ${PROJ_HOME}/tests/pre_load_models.py /

RUN cd ${PROJ_HOME}/proto && \
    protoc -I=. --python_out=. signal.proto && \
    mv signal_pb2.py / && \
    cd ${PROJ_HOME}/tests && \
    mv endpoint.py /

# disable eating up cpu resources
ENV OMP_NUM_THREADS 1

# EXPOSE 8089
WORKDIR /
RUN python pre_load_models.py

# CMD bash start.sh
CMD bash

# Test command:
# nvcc -o t1 t1.cu -cudart shared
# LD_LIBRARY_PATH=/client_bin/:$LD_LIBRARY_PATH LD_PRELOAD=/client_bin/librtclient.so ./t1