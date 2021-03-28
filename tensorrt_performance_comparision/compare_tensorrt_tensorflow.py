#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import, division, print_function, unicode_literals
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import sys
import time
import argparse
import logging
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.saved_model import tag_constants
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.python.compiler.tensorrt import trt_convert as trt

device = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(device[0], True)
tf.config.experimental.set_virtual_device_configuration(device[0],
[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])

def load_and_save_pretained_model(model_name):
    # imagenet label: https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a
    
    # load pretrained model
    if model_name == "resnet50":
        model = ResNet50(weights='imagenet')
    else:
        model = MobileNetV2(weights='imagenet')

    # save model
    save_name = "model/" + model_name + "_model_tf"
    model.save(save_name) 
    return model

def load_tensorflow_saved_model(model_name, infer_flag=None):
    # load tensorflow model
    if infer_flag is None:
        model = tf.keras.models.load_model(model_name)
    else:
        # load tensorflow graph only for interence case
        model = tf.saved_model.load(model_name, tags=[tf.saved_model.SERVING])
    return model


class PredictionImages:
    def __init__(self):
        if args.model_name == "resnet50":
            from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
        else:
            from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
        self.preprocess_input = preprocess_input
        self.decode_predictions = decode_predictions

    def process_images(self, img_path, infer_flag=None):
        # read input image
        input_img = image.load_img(img_path, target_size=(224, 224))

        # transform image for the input data
        batch_data = image.img_to_array(input_img)
        batch_data = np.expand_dims(batch_data, axis=0)
        batch_data = self.preprocess_input(batch_data)
        # in case of tensorrt, constant tensor needs to be created from tensor-like object
        batch_data = tf.constant(batch_data) if infer_flag is not None else batch_data
        pil_img = image.array_to_img(batch_data[0])
        return pil_img, batch_data

    # predict images using the model
    def predict_images(self, model, num_images, infer_flag=None):
        time_list = []
        for i in range(num_images):
            img_path = 'data/img%d.jpg'%i
            # process images
            input_img, x = self.process_images(img_path, infer_flag)

            # start timing
            start_time = time.time()
            if infer_flag is None:
                preds = model.predict(x)
            else:
                labeling = model(x)
                preds = labeling['predictions'].numpy()
            end_time = time.time()
            # end timing

            # calculate inference time
            inference_time = ( end_time - start_time ) * 1000
            time_list.append(inference_time)

            # decode the results into a list of tuples (class, description, probability)
            decoded_result = self.decode_predictions(preds, top=1)[0]

            # print out the result
            print('Image {}: {:4.1f}ms => {}'.format(i, inference_time, decoded_result))
        
        print ("median time: {} ms".format(np.median(time_list)))

        return 

class ProcessTensorRT: 
    # ## Convert tensorflow to tensorrt model
    def convert_model_from_tensorflow_to_tensorrt(self, convert_case, input_model_name, output_model_name):
        # set up conversion parameters
        if convert_case == 'fp32':
            conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
                precision_mode=trt.TrtPrecisionMode.FP32,
                max_workspace_size_bytes=8000000000)
        elif convert_case == 'fp16':
            conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
                precision_mode=trt.TrtPrecisionMode.FP16,
                max_workspace_size_bytes=8000000000)
        elif convert_case == 'int8':
            conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
                precision_mode=trt.TrtPrecisionMode.INT8, 
                max_workspace_size_bytes=8000000000, 
                use_calibration=True)

        # convert model to tensorrt (float32 or int12)
        converter = trt.TrtGraphConverterV2(input_saved_model_dir=input_model_name,
                                            conversion_params=conversion_params)
        if convert_case == 'fp32' or convert_case == 'fp16':
            converter.convert()
        elif convert_case == 'int8': # calibration is needed for int8 case
            batch_size = 1
            def calibration_input_fn():
                yield (np.random.uniform(size=(batch_size, 224, 224, 3)).astype(np.float32), )
            converter.convert(calibration_input_fn=calibration_input_fn)

        # save the converted model
        converter.save(output_saved_model_dir=output_model_name)

        return 

    def load_tensorrt_model(self, model_name):
        # load saved model
        saved_model_loaded = tf.saved_model.load(model_name, tags=[tag_constants.SERVING])
        signature_keys = list(saved_model_loaded.signatures.keys())
        infer = saved_model_loaded.signatures['serving_default']
        return infer

    # ### TF-TRT FP32 or FP16 or INT8
    def process_tftrt(self, input_tf_model, output_infer_model, case_tag):
        # convert model
        if os.path.exists(output_infer_model):
            logging.info(output_infer_model + " is exist")
        else:
            self.convert_model_from_tensorflow_to_tensorrt(case_tag, input_tf_model, output_infer_model)
        
        # load tensorrt model
        infer_model = self.load_tensorrt_model(output_infer_model)
        
        return infer_model


def main(args):
    # set up arguments
    run_case = args.case
    num_images = int(args.num_img)
    model_name = args.model_name
    
    # set up logging to file - see previous section for more details
    log_filename = "logs/output_tensorflow_{}_{}.log".format(model_name, run_case)
    os.makedirs(os.path.dirname(log_filename), exist_ok=True)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)    
    logging.basicConfig(level=logging.DEBUG, 
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M', 
                        filename=log_filename,
                        filemode='w')
    logging.getLogger('tensorflow')

    logging.info("#### start model prediction ####")
    # check data availability
    if not os.path.exists("data"):
        return logging.error('check if data is available or run download_images.py for generating sample dataset')                    

    os.makedirs(os.path.dirname("./model"), exist_ok=True)
    # check pretrained model availability
    pretrained_model_exist = "model/{}_model_tf".format(model_name)
    if os.path.exists(pretrained_model_exist):
        logging.info(pretrained_model_exist+" is exist")
    else:
        logging.info("load_and_save_pretained_model")
        load_and_save_pretained_model(model_name)

    # load tensorflow saved model
    pretrained_model_name = "model/{}_model_tf".format(model_name)
    if run_case == "tf":
        pretrained_model = load_tensorflow_saved_model(pretrained_model_name)
    elif run_case == "fp32" or run_case == "fp16" or run_case == "int8":
        pt = ProcessTensorRT()
        infer_model_name = "model/" + model_name + "_model_tftrt_" + str(run_case)
        infer_model = pt.process_tftrt(pretrained_model_name, infer_model_name, run_case)
    else:
        logging.error('please check your case argument')                    
    logging.info("model preparation is done")
    
    # predict image
    pi = PredictionImages()
    if run_case == "tf":
        pi.predict_images(pretrained_model, num_images, None)   
    elif run_case == "fp32":
        # use tensorrt model for predict image
        pi.predict_images(infer_model, num_images, "infer_fp32")
    elif run_case == "fp16":
        # use tensorrt model for predict image
        pi.predict_images(infer_model, num_images, "infer_fp16")
    elif run_case == "int8":
        # use tensorrt model for predict image
        pi.predict_images(infer_model, num_images, "infer_int8")
    else:
        logging.error('please check your case argument')                    
    
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Compare tensorrt performance')
    parser.add_argument('-model_name', help='select model: resnet50, mobilenet_v2(default)', default="mobilenet_v2", type=str, required=True)
    parser.add_argument('-case', help='select case: tf=>Tensorflow, fp32=>TF-TRT FP32, int8=>TF-TRT INT8', type=str, required=True)
    parser.add_argument('-num_img', help='set number of images to be run', type=int, required=True)
    args = parser.parse_args()
    main(args)

# python3 compare_tensorrt_tensorflow.py -model_name "mobilenet_v2" -case "tf" -num_img 10
# python3 compare_tensorrt_tensorflow.py -model_name "mobilenet_v2" -case "fp32" -num_img 10
# python3 compare_tensorrt_tensorflow.py -model_name "mobilenet_v2" -case "int8" -num_img 10

