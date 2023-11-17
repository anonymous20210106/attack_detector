import os
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
from utils.CAM_utils import *

# Detector
from ultralytics import YOLO

# Class activation map
from yolo_cam.eigen_cam import EigenCAM
from yolo_cam.utils.image import show_cam_on_image, scale_cam_image

def get_args():
    parser = argparse.ArgumentParser(description = "GradCAM")

    parser.add_argument('--img_folder',         type=str,       default=None,           help='Folder containing imgs before and after the attack')
    parser.add_argument('--model_name',         type=str,       default='n',            help='Yolov8 detection model')
    parser.add_argument('--det_threshold',      type=float,     default=0.5,            help='Detection threshold')
    parser.add_argument('--cam_threshold',      type=float,     default=0.4,            help='Detection threshold for GradCAM')
    parser.add_argument('--save_path',          type=str,       default='./visualize/', help='Save result path')

    args = parser.parse_args()

    return args

def parse_det(results):
    detections = results.boxes.xyxy.cpu()
    confs = results.boxes.conf.cpu()
    cls = results.boxes.cls.cpu()

    f = open('coco_labels.txt')
    coco_names = f.read().split('\n')


    boxes, colors, names = [], [], []

    for i in range(len(detections)):
        confidence = confs[i]
        # if confidence < 0.2:
        #     continue
        xmin = int(detections[i][0])
        ymin = int(detections[i][1])
        xmax = int(detections[i][2])
        ymax = int(detections[i][3])
        name = coco_names[int(cls[i].item())]
        category = int(cls[i])
        color = COLORS[category]

        boxes.append((xmin, ymin, xmax, ymax))
        colors.append(color)
        names.append(name)
    return boxes, colors, names, confs

def main():
    args = get_args()
    save_path           = args.save_path
    det_threshold       = args.det_threshold
    cam_threshold       = args.cam_threshold
    model_name          = 'yolov8{}.pt'.format(args.model_name)
    paths               = [os.path.join(args.img_folder, img_name) for img_name in os.listdir(args.img_folder)]
    model               = YOLO(model_name)
    target_layers       = [model.model.model[-1].cv3[0][1], model.model.model[-1].cv3[1][1], model.model.model[-1].cv3[2][1]]

    ######################################################################################################
    img = cv2.imread(paths[0])
    rgb_img = img.copy()
    img = np.float32(img) / 255

    model.conf = det_threshold
    results = model.predict(rgb_img)[0]
    det_img_0 = cv2.cvtColor(results.plot(labels=False, line_width=7), cv2.COLOR_BGR2RGB)

    boxes, colors, names, confs = parse_det(results)
    labels = results.boxes.cls

    targets = [YOLOBoxScoreTarget(labels=labels, bounding_boxes=boxes)]
    cam = EigenCAM(model,
                target_layers, 
                task='od')

    grayscale_cam = cam(rgb_img, targets=targets)
    # Take the first image in the batch:
    grayscale_cam = grayscale_cam[0, :]
    cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)
    # And lets draw the boxes again:
    image_with_bounding_boxes_0 = draw_boxes(boxes, labels, names, confs, cam_image)
    ######################################################################################################

    ######################################################################################################
    img = cv2.imread(paths[1])
    rgb_img = img.copy()
    img = np.float32(img) / 255

    model.conf = cam_threshold
    results = model.predict(rgb_img)[0]
    det_img_1 = cv2.cvtColor(results.plot(labels=False, line_width=7), cv2.COLOR_BGR2RGB)

    boxes, colors, names, confs = parse_det(results)
    labels = results.boxes.cls

    targets = [YOLOBoxScoreTarget(labels=labels, bounding_boxes=boxes)]
    cam = EigenCAM(model,
                target_layers, 
                task='od')

    grayscale_cam = cam(rgb_img, targets=targets)
    # Take the first image in the batch:
    grayscale_cam = grayscale_cam[0, :]
    cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)
    # And lets draw the boxes again:
    image_with_bounding_boxes_1 = draw_dashed_bounding_box(cam_image, boxes, labels)

    model.conf = det_threshold
    results = model.predict(rgb_img)[0]
    det_img_2 = cv2.cvtColor(results.plot(labels=False, line_width=7), cv2.COLOR_BGR2RGB)
    ######################################################################################################

    plt.figure(figsize=(30, 10), layout="compressed")

    plt.subplot(1, 4, 1)
    plt.imshow(det_img_0)
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.imshow(det_img_2)
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.imshow(image_with_bounding_boxes_0)
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.imshow(image_with_bounding_boxes_1)
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    main()