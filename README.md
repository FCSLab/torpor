# Torpor Artifact

### Introduction

Torpor is a serverless inference system that support GPU-efficient model serving through *late-binding and model swapping*. It keeps models in main memory and swaps them onto a shared pool of local GPUs when requests arrive. Torpor has been successfully integrated into Alibaba's serverless platform (refer to our paper for more details).

This repository contains the codebase for Torpor's single-node prototype. The distributed version of Torpor is tightly coupled with Alibaba's proprietary serverless platform and is not available for open-source release. The `evaluation` folder includes scripts for single-node evaluation, covering major experiments in Sections 7.1 to 7.3 of our paper.

### Get Started Instructions

#### 0. Test environment

Setup a GPU worker node (e.g., Alibaba Cloud  `ecs.gn6e-c12g1.12xlarge` )

- 48 vCPU cores
- 384 GB memory
- 4 * NVIDIA V100 GPUs each with 32 GB memory
- Ubuntu 18/20

Configurations

```
### docker and nvidia docker
sudo apt-get update
sudo apt-get install -y docker.io
systemctl start docker
systemctl enable docker
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

### python
apt-get update
apt-get install -y protobuf-compiler python3.7
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 1
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 2
pip3 install --upgrade pip
pip3 install zmq numpy protobuf==3.20.1
pip3 install requests pandas

### Torpor conf
# FIRST cp this project to local, replace {PROJ_DIR} to the actual path
cd {PROJ_DIR}/scripts/
bash compile.sh
echo max > /sys/fs/cgroup/pids/user.slice/user-0.slice/pids.max
```

#### 1. Docker images

##### Pull docker images

```
docker pull registry.cn-shenzhen.aliyuncs.com/fcslab/torpor:native-v1
docker pull registry.cn-shenzhen.aliyuncs.com/fcslab/torpor:client-v1
docker pull registry.cn-shenzhen.aliyuncs.com/fcslab/torpor:server-v1

docker tag registry.cn-shenzhen.aliyuncs.com/fcslab/torpor:native-v1 standalone-native
docker tag registry.cn-shenzhen.aliyuncs.com/fcslab/torpor:client-v1 standalone-client
docker tag registry.cn-shenzhen.aliyuncs.com/fcslab/torpor:server-v1 standalone-server
```

##### OR build from scratch

- Build base image

```
cd {PROJ_DIR}
docker build . -t standalone-base -f dockerfiles/base.Dockerfile
```

- Build standalone-native, standalone-client and standalone-server

```shell
docker build . -t standalone-native -f dockerfiles/native.Dockerfile
docker build . -t standalone-client -f dockerfiles/client.Dockerfile
docker build . -t standalone-server -f dockerfiles/server.Dockerfile
```

#### 2. Experiments

This section covers major experiments for single-node evaluation of our paper. We summarize the steps to conduct each experiment in  `evaluation` .

- Latencies with GPU remoting and model swapping (Table 4) [link](evaluation/table4)
- GPU remoting breakdown (Figure 6)  [link](evaluation/figure6)
- Model swapping breakdown (Figure 7)  [link](evaluation/figure7)
- GPU efficiency for low-frequency functions (Figure 8)  [link](evaluation/figure8)
- Cross-GPU load balancing for high-frequency functions (Figure 9)  [link](evaluation/figure9)
- Performance comparison (Figure 10)  [link](evaluation/figure10)
- Torpor with various policies (Figure 11)  [link](evaluation/figure11)

We upload logs and screenshots of relevant experiments to Google Drive, which can be found at [here](https://drive.google.com/drive/folders/1zhJh3OAfCHPx2yLyiPYTU6ttHgfcNhO_?usp=drive_link).

