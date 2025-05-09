FROM standalone-base:latest

# RUN pip3 install torch==1.11.0a0+bfe5ad2
RUN pip3 install transformers==4.27.1
RUN pip3 install pytorch-pretrained-bert==0.6.2

RUN rm -rf $PROJ_HOME
COPY . $PROJ_HOME
WORKDIR $PROJ_HOME

RUN cd ${PROJ_HOME}/proto && \
    protoc -I=. --python_out=. signal.proto && \
    mv signal_pb2.py / && \
    cd ${PROJ_HOME}/tests && \
    mv endpoint.py /

RUN cp ${PROJ_HOME}/tests/pre_load_models.py /

# EXPOSE 8089
WORKDIR /
RUN python pre_load_models.py

# CMD bash start.sh
CMD bash

# Test command:
# nvcc -o t1 t1.cu -cudart shared
# LD_LIBRARY_PATH=/client_bin/:$LD_LIBRARY_PATH LD_PRELOAD=/client_bin/librtclient.so ./t1