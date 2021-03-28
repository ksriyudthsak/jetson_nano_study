import os
import time
import argparse
import numpy as np
import cv2 as cv
import torch

from get_videocapture_model import PredictionImages, ProcessTensorRT, load_pytorch_saved_model, load_and_save_resnet50_model, download_imagenet_classes


def process_video_capture(run_case, model):    
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    img_counter = 0
    start_time = time.time()
    print ("start video")

    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        end_time = time.time()
        # if (ret == True) and (end_time - start_time < 10):
        # Display the resulting frame
        cv.imshow('frame', frame)
        if cv.waitKey(5) == ord('q'):
            break
        # read image
        # real_time_img = cv.imread(frame)

        # write image
        img_name = "./data/frame{}.JPG".format(img_counter)
        cv.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1
        # save only latest 20 frames
        img_counter = 0 if img_counter > 20 else img_counter

        # predict image
        print ("predict image")
        pi = PredictionImages()
        if run_case == "torch":
            pi.predict_images(model, img_name, None)
        elif run_case == "torch2trt":
            pi.predict_images(model, img_name, "infer_torch2trt")

        # else:
        #     break
    print ("end video")
    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()
    return 

def main(arg):
    run_case = args.case
    model_name = "resnet50"

    # check model and data directory
    os.makedirs(os.path.dirname("model/"), exist_ok=True)
    os.makedirs(os.path.dirname("data/"), exist_ok=True)

    # check label
    if not os.path.exists("data/imagenet_classes.txt"):
        download_imagenet_classes()

    # check pretrained model availability
    pretrained_model_exist = "model/{}_model_torch".format(model_name)
    if not os.path.exists(pretrained_model_exist):
        load_and_save_resnet50_model()
    
    # load pytorch saved model
    print ("load a model")
    if run_case == "torch":
        print ("torch model")
        model = load_pytorch_saved_model('./model/resnet50_model_torch')
    elif run_case == "torch2trt":
        print ("torch2trt model")
        pt = ProcessTensorRT()        
        model = pt.process_tftrt('./model/resnet50_model_torch', "./model/resnet50_pytorch_trt.pth")
    else:
        model = None
        print ("no model")

    print ("start real-time video")
    process_video_capture(run_case, model)
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Real-time detection')
    parser.add_argument('-case', help='select case: torch=>Pytorch, ', type=str, required=True)
    args = parser.parse_args()
    main(args)


# # python3 run_realtime_videocapture.py -case "torch"
# # python3 run_realtime_videocapture.py -case "torch2trt"
