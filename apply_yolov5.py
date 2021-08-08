# These codes are inspiring and editing from regional yolov5 example code (https://github.com/ultralytics/yolov5.git)
# And before you run this script you need to install all the yolov5 dependency as well.
# One of the yolov5 dependency is Pytorch and it might not able to be installed by using PIP
# To know which way is best for you, you should go to (https://pytorch.org/) to found out.

import time
from pathlib import Path
import cv2
import torch
from numpy import random
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, strip_optimizer, set_logging)
from utils.torch_utils import select_device, load_classifier


def detect(source, output=None, weights='yolov5x.pt', conf_thres=0.6, iou_thres=0.7):
    # Initialize
    set_logging()
    device = select_device("")
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(960, s=model.stride.max())  # check img_size (default value is 640)
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

    time1 = time.time()
    # Set Dataloader
    dataset = LoadImages(source, img_size=imgsz)

    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # Inference
        pred = model(img, augment=False)[0]
        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, agnostic=False,classes=None)
        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)
        # Process detections
        print()
        for i, det in enumerate(pred):  # detections per image
            p, s, im0 = path, '', im0s
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    start_point=(int(xyxy[0]), int(xyxy[1]))
                    end_point = (int(xyxy[2]), int(xyxy[3]))
                    print((start_point, end_point),names[int(cls)],int(conf*1000)/1000)
                    if output != None:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
            else:
                print("nothing found here")
            # Print time (inference + NMS)
            #print('%sDone. (%.3fs)' % (s, t2 - t1))
            # Save results (image with detections)
            if output != None:
                cv2.imwrite(str(Path(output) / Path(p).name), im0)
    if output != None:
        print('Results saved to %s' % Path(output))
    print("used time:",int((time.time() - time1)*100)/100,"s")
    #print('Done. (%.3fs)' % (time.time() - t0))


with torch.no_grad():
    detect('DeFacTo.jpg')
    