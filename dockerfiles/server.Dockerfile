FROM standalone-base:latest

RUN rm -rf $PROJ_HOME
COPY . $PROJ_HOME
WORKDIR $PROJ_HOME

RUN cd ${PROJ_HOME} && \
    rm -rf build && \
    mkdir build && \
    cd build && \
    mv /usr/local/bin/protoc* /opt/conda/lib/python3.8/site-packages/torch/bin/ && \
    cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_TARGET:STRING=server -DCMAKE_PREFIX_PATH='/opt/conda/lib/python3.8/site-packages/torch' .. && \
    make

RUN rm -rf /server_bin && \
    mkdir /server_bin && \
    cp ${PROJ_HOME}/build/target/server /server_bin && \
    cp ${PROJ_HOME}/build/src/server/libwapper.so /server_bin && \
    cd /server_bin && \
    # ldd server | grep "=> /" | awk '{print $3}' | xargs -I '{}' cp -v '{}' . && \
    echo "LD_LIBRARY_PATH=/usr/local/cuda/lib64 LD_PRELOAD=./libwapper.so ./server" > start.sh && \
    chmod +x start.sh
    
# RUN cd ${PROJ_HOME}/tests && \
#     mv cv_endpoint.py /

# EXPOSE 8088
WORKDIR /server_bin
CMD bash start.sh