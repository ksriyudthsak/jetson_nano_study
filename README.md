# Jetson Nano Study

This is for my study on edge AI (using Jetson Nano 2GB)


## TensorRT Performance Comparison
* All cases are run under docker container (See how to set up and run the container below)
* Model, logs, data folders will be automatically generated at the first run

### 1. download images
* download_images.py

### 2. run comparisons (three main cases are available)
* compare_tensorrt_pytorch.py
    * example 1: <pre>python3 compare_tensorrt_pytorch.py -model_name "resnet50" -case "torch" -num_img 10
    * example 2: <pre>python3 compare_tensorrt_pytorch.py -model_name "resnet50" -case "torch2trt" -num_img 10
* compare_tensorrt_native.py</pre>
    * example <pre>python3 compare_tensorrt_native.py -model_name "resnet50" -case "torch_onnx" -num_img 10
* compare_tensorrt_tensorflow.py</pre>
    * example (work in Google Colab but too heavy for Jetson Nano 2GB) <pre>python3 compare_tensorrt_tensorflow.py -model_name "resnet50" -case "int8" -num_img 10</pre>

## Real-time Image Classification
* run_realtime_videocapture.py
    * example <pre>python3 run_realtime_videocapture.py -case "torch2trt"</pre>


## Docker Container (for PyTorch or TensorRT based models)

<pre><code># pull ml docker
docker pull nvcr.io/nvidia/l4t-ml:r32.4.4-py3

# run docker container (for installation of other libraries)
docker run -it --gpus all --runtime nvidia --network host -v /home/ks/Documents:/mnt/ml nvcr.io/nvidia/l4t-ml:r32.4.4-py3 bin/bash

# install opencv inside the container
apt-get update
apt-get install libopencv-dev
apt-get install python3-opencv 

# install  torch2trt inside the container
git clone https://github.com/NVIDIA-AI-IOT/torch2trt
cd torch2trt
python3 setup.py install

# exit the container
exit 

# check the container history
docker ps -a

# commit the container to a new image
# command: docker commit [CONTAINER_ID] [new_image_name]
docker commit b6054fa07233 l4t-ml-cv-trt

# give an access to server from all host (for sharing display)
xhost +

# run the l4t-ml + cv2 + torch2trt container 
docker run -it --rm --gpus all --runtime nvidia --network host -v /home/ks/Documents:/mnt/ml --device /dev/video0 -e DISPLAY=$DISPLAY l4t-ml-cv-trt:latest
</code></pre>

## Docker Container (for tensorflow models)

<pre><code># pull tensorflow docker
docker pull nvcr.io/nvidia/l4t-tensorflow:r32.4.4-tf2.3-py3

# run docker container
docker run -it --rm --gpus all --runtime nvidia --network host -v /home/ks/Documents:/mnt/ml nvcr.io/nvidia/l4t-tensorflow:r32.4.4-tf2.3-py3

# install jupyter notebook in docker container (optional)
pip3 install --upgrade pip
pip3 install jupyter lab

# run jupyter notebook (optional)
jupyter notebook --ip=0.0.0.0 --allow-root
</code></pre>
