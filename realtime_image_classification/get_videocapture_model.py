#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import, division, print_function, unicode_literals
import os
import time
import urllib.request
import argparse
import logging

import numpy as np

import torch
from torchvision import models
from torchvision import transforms
from PIL import Image

from torch2trt import torch2trt, TRTModule


def load_and_save_resnet50_model():
    # load pretrained model
    resnet50_model = models.resnet50(pretrained=True)

    # save model
    torch.save(resnet50_model, './model/resnet50_model_torch')
    return
   
def load_pytorch_saved_model(model_name):
    # load pytorch model
    model = torch.load(model_name)

    # set model to evaluation model and sent to GPU
    if torch.cuda.is_available():
        model = model.eval().cuda()
    return model

def download_imagenet_classes():
    imagenet_classes = "https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw/238f720ff059c1f82f368259d1ca4ffa5dd8f9f5/imagenet1000_clsidx_to_labels.txt"
    with urllib.request.urlopen(imagenet_classes) as u:
        with open('./data/imagenet_classes.txt', 'wb') as o:
            o.write(u.read())


class PredictionImages:
    def process_images(self, img_path):
        # read input image
        input_img = Image.open(img_path)

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
        if torch.cuda.is_available():
            batch_data = batch_data.cuda()
        return input_img, batch_data

    def decode_predictions_top5(self, out, top=5):
        # Load the file containing the 1,000 labels for the ImageNet dataset classes
        # imagenet label: https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a

        with open('./data/imagenet_classes.txt') as f:
            labels = [line.strip() for line in f.readlines()]

        # find the index (tensor) corresponding to the maximum score in the out tensor. 
        _, index = torch.max(out, 1)

        # find the score in terms of percentage by using torch.nn.functional.softmax function
        # which normalizes the output to range [0,1] and multiplying by 100
        percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100

        # # print the name along with score of the object identified by the model
        # print(labels[index[0]], percentage[index[0]].item())

        # print the top 5 scores along with the image label. Sort function is invoked on the torch to sort the scores. 
        _, indices = torch.sort(out, descending=True)
        [(labels[idx], percentage[idx].item()) for idx in indices[0][:top]]
        top5 = ([(labels[idx], percentage[idx].item()) for idx in indices[0][:top]])
        return top5

    # predict images using the model
    def predict_images(self, model, real_time_img, infer_flag=None):
        # fig, axes = plt.subplots(nrows=int(num_images/2), ncols=2)
        input_img, x = self.process_images(real_time_img)

        # start timing # ========
        start_time = time.time()
        if infer_flag is None:
            preds = model(x)
        else:
            preds = model(x)
            # check the output against PyTorch
            # print(torch.max(torch.abs(y - y_trt)))

        end_time = time.time()
        # end timing # ========

        # calculate inference time
        inference_time = ( end_time - start_time ) * 1000

        # decode the results into a list of tuples (class, description, probability)
        decoded_result = self.decode_predictions_top5(preds, 3)

        # print out the result
        print('Image: {:4.1f}ms => {}'.format(inference_time, decoded_result))
        return 


class ProcessTensorRT:
    def process_tftrt(self, input_model, output_infer_model):
        if os.path.exists(output_infer_model):            
            logging.info("resnet50_pytorch_trt.pth is exist")
            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(output_infer_model)) 
        else:
            # load pretrained model
            resnet50_model = load_pytorch_saved_model(input_model)
            # convert to TensorRT feeding sample data as input
            x = torch.ones((1, 3, 224, 224)).cuda()
            model_trt = torch2trt(resnet50_model, [x])
            
            # save and load
            torch.save(model_trt.state_dict(), output_infer_model)
        return model_trt
