#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import, division, print_function, unicode_literals
import os
import time
import argparse
import logging
import numpy as np

import torch
from torchvision import models
from torchvision import transforms
from PIL import Image


def load_and_save_pretained_model(model_name):
    # load pretrained model (default = resnet50)
    if model_name == "resnet101":
        model = models.resnet101(pretrained=True)
    elif model_name == "mobilenet_v2":
        model = models.mobilenet_v2(pretrained=True)
    elif model_name == "resnet152":
        model = models.resnet152(pretrained=True)
    elif model_name == "inception_v3":
        model = models.inception_v3(pretrained=True)
    elif model_name == "wide_resnet50_2":
        model = models.wide_resnet50_2(pretrained=True)
    else:
        model = models.resnet50(pretrained=True)

    # save model
    filename = 'model/{}_model_torch'.format(model_name)
    torch.save(model, filename)
    return
   
def load_pytorch_saved_model(model_name):
    # load pytorch model
    model = torch.load(model_name)

    # set model to evaluation model and sent to GPU
    model = model.eval().cuda()
    return model


class PredictionImages:
    def process_images(self, img_path):
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
        return input_data, batch_data.cuda()

    def decode_predictions_top5(self, out, top=1):
        # Load the file containing the 1,000 labels for the ImageNet dataset classes
        # imagenet label: https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a

        with open('data/imagenet_classes.txt') as f:
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
    def predict_images(self, model, num_images, infer_flag=None):
        time_list = []
        for i in range(num_images):
            img_path = 'data/img%d.jpg'%i
            input_img, x = self.process_images(img_path)

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
            time_list.append(inference_time)
            
            # decode the results into a list of tuples (class, description, probability)
            decoded_result = self.decode_predictions_top5(preds, 1)

            # print out the result
            print('Image {}: {:4.1f}ms => {}'.format(i, inference_time, decoded_result))
        print ("median time: {} ms".format(np.median(time_list)))
        return 


class ProcessTensorRT:
    # load torch2trt in this class so that this file can run pytorch model even in an environment which does not contain tensortrt (for simple testing in google colaboratory)
    def __init__(self):
        from torch2trt import torch2trt, TRTModule
        self.torch2trt = torch2trt
        self.TRTModule = TRTModule

    def process_tftrt(self, input_model, output_infer_model):
        if os.path.exists(output_infer_model):            
            logging.info(output_infer_model + ".pth is exist")
            model_trt = self.TRTModule()
            model_trt.load_state_dict(torch.load(output_infer_model)) 
        else:
            # load pretrained model
            model = load_pytorch_saved_model(input_model)

            # convert to TensorRT feeding sample data as input
            x = torch.ones((1, 3, 224, 224)).cuda()
            model_trt = self.torch2trt(model, [x])
            
            # save and load
            torch.save(model_trt.state_dict(), output_infer_model)
        return model_trt


def main(args):
    # set up arguments
    run_case = args.case
    num_images = int(args.num_img)
    model_name = args.model_name

    # set up logging to file - see previous section for more details
    log_filename = "logs/output_torch_{}_{}.log".format(model_name, run_case)
    os.makedirs(os.path.dirname(log_filename), exist_ok=True)
    logging.basicConfig(level=logging.DEBUG, 
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M', 
                        filename=log_filename,
                        filemode='w')
    logging.getLogger('trt')

    # check model directory
    os.makedirs(os.path.dirname("model/"), exist_ok=True)
    
    # check data availability
    logging.info("check if data is available")
    logging.info("check data")
    if not os.path.exists("data"):
        return logging.error('check if data is available or run download_images.py for generating sample dataset')                    

    # check pretrained model availability
    pretrained_model_exist = "model/{}_model_torch".format(model_name)
    if os.path.exists(pretrained_model_exist):
        logging.info(pretrained_model_exist+".pth is exist")
    else:
        logging.info("load_and_save_pretained_model")
        load_and_save_pretained_model(model_name)
    
    # load tensorflow saved model
    if run_case == "torch":
        pretrained_model = load_pytorch_saved_model('model/{}_model_torch'.format(model_name))
    elif run_case == "torch2trt":
        pt = ProcessTensorRT()        
        infer_model = pt.process_tftrt('model/{}_model_torch'.format(model_name), "model/{}_torch_trt.pth".format(model_name))
    else:
        logging.error('please check your case argument')                    
    logging.info("model preparation is done")

    # predict image
    pi = PredictionImages()	
    if run_case == "torch":
        pi.predict_images(pretrained_model, num_images, None)
    elif run_case == "torch2trt":
        pi.predict_images(infer_model, num_images, "infer_torch2trt")
    else:
        logging.error('please check your case argument')                    

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Compare tensorrt performance')
    parser.add_argument('-model_name', help='select model: resnet50(default), mobilenet_v2, resnet152, inception_v3, wide_resnet50_2', default="resnet50", type=str, required=True)
    parser.add_argument('-case', help='select case: torch=>Pytorch, ', type=str, required=True)
    parser.add_argument('-num_img', help='set number of images to be run', type=int, required=True)
    args = parser.parse_args()
    main(args)

# #### install torch2trt
# git clone https://github.com/NVIDIA-AI-IOT/torch2trt
# cd torch2trt
# python3 setup.py install

# #### run comparison
# python3 compare_tensorrt_pytorch.py -model_name "resnet50" -case "torch" -num_img 10
# python3 compare_tensorrt_pytorch.py -model_name "resnet50" -case "torch2trt" -num_img 10
