# How to start LVI-SAM with Docker

Docker makes it easy to configure the configuration required to run LVI-SAM.  
Before you start LVI-SAM with docker, you should install [docker](https://www.docker.com/) and [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) in your PC.  

## 1. Make LVI-SAM Docker image  

There are two ways to make docker image.  

### 1-1. Pull docker image in dockerhub  

You can easy to make docker image to pull it in [dockerhub](https://hub.docker.com/), this image is made from the Dockerfile in `docker/` folder.  

```
docker pull tyoung96/lvi_sam:1.0
```

### 1-2 Build docker image with Dockerfile  

You could also make docker image directly with provieded Dockerfile.  

Move the terminal path to `/docker` and execute the following command.  

```
docker build -t lvi_sam:1.0 .
```

`lvi_sam:1.0` is just example of this docker image, you can replace it with the image name you want.  

After the image is created, you can execute `docker images` command to view the following results from the terminal.

```
REPOSITORY                  TAG                    IMAGE ID       CREATED             SIZE
tyoung96/lvi_sam            1.0                    ece4f57ca14b   48 minutes ago      7.99GB
```

### Docker image information  

||Version|  
|:---:|:---:|  
|Ubuntu|18.04| 
|ROS|Melodic|     
|CUDA|11.2|  
|OpenCV|3.2.0|
|PCL|1.8|  
|Eigen|3.3.4|  
|ceres-solver|1.14.0|  
|GTSAM|4.0.2|  

## 2. Make LVI-SAM docker container  

When you create a docker container, you need several options to use the GUI and share folders.  

First, you should enter the command below in the local terminal to enable docker to communicate with Xserver on the host.  

```
xhost +local:docker
```

After that, make your own container with the command below.  

```
nvidia-docker run --privileged -it \
           -e NVIDIA_DRIVER_CAPABILITIES=all \
           -e NVIDIA_VISIBLE_DEVICES=all \
           --volume=${LVI-SAM_repo_root}:/home/catkin_ws/src \
           --volume=/tmp/.X11-unix:/tmp/.X11-unix:rw \
           --net=host \
           --ipc=host \
           --shm-size=1gb \
           --name=${docker container name} \
           --env="DISPLAY=$DISPLAY" \
           ${docker image} /bin/bash
```   

⚠️ **You should change {LVI-SAM_repo_root}, {docker container name}, {docker image} to suit your environment.**  

For example,  
```
nvidia-docker run --privileged -it \
           -e NVIDIA_DRIVER_CAPABILITIES=all \
           -e NVIDIA_VISIBLE_DEVICES=all \
           --volume=/home/taeyoung/Desktop/LVI-SAM:/home/catkin_ws/src \
           --volume=/tmp/.X11-unix:/tmp/.X11-unix:rw \
           --net=host \
           --ipc=host \
           --shm-size=1gb \
           --name=lvi-sam \
           --env="DISPLAY=$DISPLAY" \
           tyoung96/lvi_sam:1.0 /bin/bash
```

If you have successfully created the docker container, the terminal output will be similar to the below.  

```
================Docker Env Ready================
root@taeyoung-cilab:/home/catkin_ws#
```  


These docker tutorial is tested on ubuntu 18.04 and may not be applied to arm platforms such as NVIDIA Jetson. In addition, this docker tutorial was used to execute the LVI-SAM with a bagfile, and if the actual sensor is used, it needs to be modified to create a docker container.  

## Reference  
 - [electech6/LVI-SAM_detailed_comments](https://github.com/electech6/LVI-SAM_detailed_comments)  
 - [engcang/SLAM-application](https://github.com/engcang/SLAM-application)  
 - [FasterLIO docker folder](https://github.com/gaoxiang12/faster-lio/tree/main/docker)  

