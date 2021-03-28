#!/usr/bin/env python
# coding: utf-8

import os
import time
import argparse
import numpy as np
import logging

import torch
from torchvision import models
from torchvision import transforms
from PIL import Image

import tensorrt as trt
import engine as eng
from onnx import ModelProto

import pycuda.driver as cuda
import pycuda.autoinit 

# #### Directly convert keras model to TensorRT
def convert_tf_to_onnx():
    # tf2onnx module needs to be installed in order to use this function
    import onnx
    import tf2onnx
    img0, x = process_images('./data/img0.jpg')
    model_proto, external_tensor_storage = tf2onnx.convert.from_keras(model, input_signature=[tf.TensorSpec(shape=x.shape, dtype=tf.float32)])
    onnx.save(model_proto,'./resnet50_model_tf/resnet50_tf_onnx.onnx"')
    # ! python3 -m tf2onnx.convert --saved-model "./resnet50_model_tf" --output "./resnet50_model_tf/resnet50_tf_onnx.onnx"
    return


def convert_torch_to_onnx(model_name):
    # https://pytorch.org/docs/stable/onnx.html
    size= 224
    # Pre-process the image and convert into a tensor
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225]),
    ])
    if model_name == "mobilenet_v2":
        trained_model = models.mobilenet_v2(pretrained=True)
    else:
        trained_model = models.resnet50(pretrained=True)

    img_path = './data/img0.jpg'
    img = Image.open(img_path).convert('RGB')
    dummy_input = transform(img).unsqueeze(0)
    onnx_output = 'model/' + model_name + '_torch_onnx.onnx'
    torch.onnx.export(trained_model, dummy_input, onnx_output)
    # torch.onnx.export(trained_model, dummy_input, onnx_output, dynamic_axes={"actual_input_1":{0:"batch_size"}, "output2":{0:"batch_size"}})

    # visualise onnx model
    # ! python3 -m pip install netron
    # http://localhost:8080/ 
    return


def process_images(img_path):
    # read input image
    input_img = Image.open(img_path).convert('RGB')

    # transform image for the input data
    preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )])  
    input_data = preprocess(input_img)
    batch_data = torch.unsqueeze(input_data, 0)
    return input_img, batch_data#.cuda()

def decode_predictions(out):
    # Load the file containing the 1,000 labels for the ImageNet dataset classes
    # imagenet label: https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a

    with open('./data/imagenet_classes.txt') as f:
        labels = [line.strip() for line in f.readlines()]

    # find the index (tensor) corresponding to the maximum score in the out tensor. 
    out_list = out.tolist()
    max_value = max(out_list)
    max_index = out_list.index(max_value)

    return labels[max_index]

def predict_images(num_images, engine, h_input_1, d_input_1, h_output, d_output, stream):

    height, width = 224, 224
    time_list = []
    for i in range(num_images):
        img_path = 'data/img%d.jpg'%i
        input_img, batch_data = process_images(img_path)
        pics_1 = batch_data
        batch_size = 1
        # start timing # ========
        # start_time = time.time()
        out, inference_time = do_inference(engine, pics_1, h_input_1, d_input_1, h_output, d_output, stream, batch_size, height, width)
        # end_time = time.time()
        # end timing # ========
        
        # calculate inference time
        # inference_time = ( end_time - start_time ) * 1000    
        time_list.append(inference_time)
        
        # decode the result
        decoded_result = decode_predictions(out)
        print('Image {}: {:4.1f}ms => {}'.format(i, inference_time, decoded_result))
    print ("median time: {} ms".format(np.median(time_list)))
    return

##############################

# build engine using onnx model
def build_engine_from_onnx(onnx_path, engine_name, batch_size, TRT_LOGGER):
    model = ModelProto()
    with open(onnx_path, "rb") as f:
        model.ParseFromString(f.read())

    d0 = model.graph.input[0].type.tensor_type.shape.dim[1].dim_value
    d1 = model.graph.input[0].type.tensor_type.shape.dim[2].dim_value
    d2 = model.graph.input[0].type.tensor_type.shape.dim[3].dim_value
    shape = [batch_size , d0, d1 ,d2]
    engine = eng.build_engine(TRT_LOGGER, onnx_path, shape= shape)
    eng.save_engine(engine, engine_name) 
    return engine

# Running inference from the TensorRT engine:
def allocate_buffers(engine, batch_size, data_type):

    """
    This is the function to allocate buffers for input and output in the device
    Args:
        engine : The path to the TensorRT engine. 
        batch_size : The batch size for execution time.
        data_type: The type of the data for input and output, for example trt.float32. 

    Output:
        h_input_1: Input in the host.
        d_input_1: Input in the device. 
        h_output_1: Output in the host. 
        d_output_1: Output in the device. 
        stream: CUDA stream.

    """

    # Determine dimensions and create page-locked memory buffers (which won't be swapped to disk) to hold host inputs/outputs.
    h_input_1 = cuda.pagelocked_empty(batch_size * trt.volume(engine.get_binding_shape(0)), dtype=trt.nptype(data_type))
    h_output = cuda.pagelocked_empty(batch_size * trt.volume(engine.get_binding_shape(1)), dtype=trt.nptype(data_type))

    # Allocate device memory for inputs and outputs.
    d_input_1 = cuda.mem_alloc(h_input_1.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)

    # Create a stream in which to copy inputs/outputs and run inference.
    stream = cuda.Stream()
    return h_input_1, d_input_1, h_output, d_output, stream 

def load_images_to_buffer(pics, pagelocked_buffer):
    preprocessed = np.asarray(pics).ravel()
    np.copyto(pagelocked_buffer, preprocessed) 

def do_inference(engine, pics_1, h_input_1, d_input_1, h_output, d_output, stream, batch_size, height, width):
    """
    This is the function to run the inference
    Args:
        engine : Path to the TensorRT engine 
        pics_1 : Input images to the model.  
        h_input_1: Input in the host         
        d_input_1: Input in the device 
        h_output_1: Output in the host 
        d_output_1: Output in the device 
        stream: CUDA stream
        batch_size : Batch size for execution time
        height: Height of the output image
        width: Width of the output image

    Output:
        The list of output images

    """

    load_images_to_buffer(pics_1, h_input_1)

    with engine.create_execution_context() as context:
        # Transfer input data to the GPU.
        cuda.memcpy_htod_async(d_input_1, h_input_1, stream)

        # Run inference.
        # start timing # ========
        start_time = time.time()
        # context.profiler = trt.Profiler()
        context.execute(batch_size=1, bindings=[int(d_input_1), int(d_output)])
        end_time = time.time()
        # end timing # ========
        # calculate inference time
        inference_time = ( end_time - start_time ) * 1000    
        # print ("inference time: {}".format(inference_time))

        # Transfer predictions back from the GPU.
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        # Synchronize the stream
        stream.synchronize()
        # Return the host output.
        # out = h_output.reshape((batch_size, -1, height, width))
        out = h_output
        return out, inference_time

def main():
    # set up arguments
    run_case = args.case
    num_images = int(args.num_img)
    model_name = args.model_name

    # set up logging to file - see previous section for more details
    log_filename = "logs/output_tensorrt_{}_{}.log".format(model_name, run_case)
    os.makedirs(os.path.dirname(log_filename), exist_ok=True)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)    
    logging.basicConfig(level=logging.DEBUG, 
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M', 
                        filename=log_filename,
                        filemode='w')
    logging.getLogger('tensorrt')
    logging.info("#### start model prediction ####")
 
    # logger to capture errors, warnings, and other information during the build and inference phases
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    trt_runtime = trt.Runtime(TRT_LOGGER)

    # check model directory
    os.makedirs(os.path.dirname("model/"), exist_ok=True)

    # check data availability
    logging.info("check if data is available")
    logging.info("check data")
    if not os.path.exists("data"):
        return logging.error('check if data is available or run download_images.py for generating sample dataset')                    

    batch_size = 1
    # running inference from tensorrt
    logging.info("check engine")
    engine_name = "model/" + model_name + "_" + run_case + ".plan"
    if os.path.exists(engine_name):
        engine = eng.load_engine(trt_runtime, engine_name)
    else:
        onnx_path = "model/" + model_name + "_" + run_case + ".onnx"
        if not os.path.exists(onnx_path):
            logging.info("convert_torch_to_onnx")
            convert_torch_to_onnx(model_name)
        logging.info("build_engine_from_onnx")
        engine = build_engine_from_onnx(onnx_path, engine_name, batch_size, TRT_LOGGER)

    logging.info("start interference")
    data_type = trt.float32
    logging.info("allocate_buffers")
    h_input_1, d_input_1, h_output, d_output, stream = allocate_buffers(engine, batch_size, data_type)

    # predict images
    logging.info("do_inference")
    predict_images(num_images, engine, h_input_1, d_input_1, h_output, d_output, stream)

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'TensorRT')
    parser.add_argument('-model_name', help='select model: resnet50(default), mobilenet_v2', default="resnet50", type=str, required=True)
    parser.add_argument('-case', help='select case: ', type=str, required=True)
    parser.add_argument('-num_img', help='set number of images to be run', type=int, required=True)
    args = parser.parse_args()
    main()

# python3 compare_tensorrt_native.py -model_name "resnet50" -case "torch_onnx" -num_img 10
# python3 compare_tensorrt_native.py -model_name "resnet50" -case "tf_onnx" -num_img 10
