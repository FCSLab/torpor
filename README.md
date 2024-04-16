## Standalone system

### 1. Docker images

#### pull server and client images

```
docker login --username=fc_cn@test.aliyunid.com registry.cn-shanghai.aliyuncs.com
password: Serverless123@aliyun

docker pull registry.cn-shanghai.aliyuncs.com/fc-demo2/gpu-swap-standalone-base:client-2
docker pull registry.cn-shanghai.aliyuncs.com/fc-demo2/gpu-swap-standalone-base:server-2

docker tag registry.cn-shanghai.aliyuncs.com/fc-demo2/gpu-swap-standalone-base:client-2 standalone-client
docker tag registry.cn-shanghai.aliyuncs.com/fc-demo2/gpu-swap-standalone-base:server-2 standalone-server

```

#### or build from scratch

- Build base image

```
cd standalone
docker build . -t standalone-base -f dockerfiles/base.Dockerfile

```
- Build CUDA server and client

```
docker build . -t standalone-server -f dockerfiles/server.Dockerfile
docker build . -t standalone-client -f dockerfiles/client.Dockerfile

```

### 2. Run system on Ubuntu VM

#### init and config

- nvidia docker

```
sudo apt-get update
sudo apt-get install -y docker.io
systemctl start docker
systemctl enable docker

distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

```

- python

```
apt-get update
apt-get install -y protobuf-compiler python3.7

sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 1
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 2

# update pip
pip3 install --upgrade pip

# install packages
pip3 install zmq numpy protobuf==3.20.1

```

- config system

```
# cp this project to local, and cd the standalone dir

cd scripts/
bash compile.sh

echo max > /sys/fs/cgroup/pids/user.slice/user-0.slice/pids.max

```

#### run tests

- launch CUDA server

```
docker run --gpus all --rm --network=host --ipc=host -v /dev/shm/ipc:/cuda -e MEM_LIMIT_IN_GB=25 -e IO_THREAD_NUM=4 -it  standalone-server  bash start.sh

```

- run test router

```
# cd standalone
cd tests/ 
bash ../scripts/compile.sh

python3 router_seq.py

# clear up test
docker ps -aq --filter ancestor=standalone-client | xargs docker stop


```
