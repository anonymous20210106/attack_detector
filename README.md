<!-- PAPER TITLE -->

# [ACCV '24] Distortion-Aware Adversarial Attacks on Bounding Boxes of Object Detectors

<!-- TABLE OF CONTENTS -->

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#authors">Authors</a></li>
    <li><a href="#abstract">Abstract</a></li>
    <li><a href="#key-properties">Key Properties</a></li>
    <li><a href="#cross-model-transferability">Cross-Model Transferability</a></li>
    <li><a href="#ultralytics-yolov8-attack">Ultralytics YOLOv8 Attack</a></li>
    <li><a href="#mmdetection-attack">MMDetection Attack</a></li>
    <li><a href="#citing">Citing</a></li>
  </ol>
</details>

## Authors

<ol>
    <li><a href="">Anonymous Author 1</a></li>
    <li><a href="">Anonymous Author 2</a></li>
    <li><a href="">Anonymous Author 3</a></li>
</ol>

<!-- ABSTRACT -->

## Abstract

Deep learning-based object detection has become ubiquitous in the last decade due to its high accuracy in many real-world applications. With this growing trend, these models are interested in being attacked by adversaries, with most of the results being on classifiers, which do not match the context of practical object detection. In this work, we propose a novel method to fool object detectors, expose the vulnerability of state-of-the-art detectors, and promote later works to build more robust detectors to adversarial examples. Our method aims to generate adversarial images by perturbing object confidence scores during training, which is crucial in predicting confidence for each class in the testing phase. Herein, we provide a more intuitive technique to embed additive noises based on detected objects' masks and the training loss with distortion control over the original image by leveraging the gradient of iterative images. To verify the proposed method, we perform adversarial attacks against different object detectors, including the most recent state-of-the-art models like YOLOv8, Faster R-CNN, RetinaNet, and Swin Transformer. We also evaluate our technique on MS COCO 2017 and PASCAL VOC 2012 datasets and analyze the trade-off between success attack rate and image distortion. Our experiments show that the achievable success attack rate is up to $100$\% and up to $98$\% when performing white-box and black-box attacks, respectively. The demo is available at [YouTube](https://youtu.be/y_sQqECMJIk).

<p align="center">
   <img src="images/demo.gif" data-canonical-src="images/demo.gif" width="600"/><br/>
   <i>Side-by-side video illustrates original (left) and disabled (right) cameras.</i>
 </p>

<!-- KEY PROPERTIES -->

## Key Properties

Comparisons between [DAG](https://openaccess.thecvf.com/content_ICCV_2017/papers/Xie_Adversarial_Examples_for_ICCV_2017_paper.pdf), [UEA](https://arxiv.org/pdf/1811.12641.pdf),
and our method for adversarial attacks on object detectors.

<center>

|               | [DAG](https://openaccess.thecvf.com/content_ICCV_2017/papers/Xie_Adversarial_Examples_for_ICCV_2017_paper.pdf) | [UEA](https://arxiv.org/pdf/1811.12641.pdf) | Ours
| :---------------: | :------: | :---: | :---: |
| iterative added noises | &check; | &cross; | &check; |
| mostly imperceptible to human eyes | &check; | &check; | &check; |
| distortion awareness | &cross; | &cross; | &check; |
| stable transferability to other backbones | &cross; | &cross; | &check; |
| consistent with detection algorithms | &cross; | &cross; | &check; |

</center>

<!-- CROSS-MODEL TRANSFERABILITY -->

## Cross-Model Transferability

### Qualitative Results

<p align="center">
   <img src="images/qualitative_results.png" data-canonical-src="images/demo.gif" width="800"/><br/>
   Qualitative results of the generated adversarial image against YOLOv8x that perturbs other detection models, including YOLO's versions, Faster R-CNN, RetinaNet, and Swin-T, at confidence thresholds of 0.50.
 </p>


### Quantitative Results

#### Quantitative Results in mAP

<center>

| Added Perturbation| YOLOv8n | YOLOv8s | YOLOv8m | YOLOv8l | YOLOv8x | Faster R-CNN | RetinaNet | Swin-T |
| :---------------: | :------: | :----: | :------: | :----: |:------: | :----: | :------: | :----: |
| **None** (baseline) | 25.04 | 33.26 | 36.98 | 38.94 | 40.02 | 27.90 | 22.90 | 32.47 |
| **YOLOv8n** | 0.06 | 18.12 | 25.19 | 28.25 | 29.52 | 13.57 | 10.69 |  17.22 |
| **YOLOv8s** | 3.32 | 0.03 | 16.71 | 20.68 | 22.45 | 9.68 | 7.31 | 13.66 | 
| **YOLOv8m** | 2.21 | 4.35 | 0.02 | 13.12 | 15.32 | 7.03 | 5.01 | 10.69 |
| **YOLOv8l** | 1.69 | 3.52 | 6.90 | 0.02 | 11.37 | 6.36 | 4.35 | 10.18 |
| **YOLOv8x** | 1.42 | 2.93 | 5.47 | 6.50 | 0.05 | 5.32 | 3.58 | 8.57 |
| **Faster R-CNN** | 3.86 | 6.96 | 10.51 | 13.09 | 13.96 | 0.10 | 0.60 | 12.70 |
| **RetinaNet** | 6.01 | 9.99 | 14.22 | 17.01 | 18.01 | 2.10 | 0.30 | 16.00 |
| **Swin-T** | 2.98 | 5.83 | 9.49 | 12.42 | 14.50 | 11.30 | 8.70 | 0.10 |

</center>

<p align="center">
    Table for cross-model transferability among commonly used detection models (in mAP) of various-sized YOLO's, Faster R-CNN, RetinaNet, and Swin Transformer, at confidence thresholds of 0.50, validating on <b>MS COCO 2017 validation set</b>.
</p>

<center>

| Added Perturbation| YOLOv8n | YOLOv8s | YOLOv8m | YOLOv8l | YOLOv8x | Faster R-CNN | RetinaNet | Swin-T |
| :---------------: | :------: | :----: | :------: | :----: |:------: | :----: | :------: | :----: |
| **None** (baseline) | 45.15 | 54.45 | 60.80 | 63.47 | 64.00 | 46.13 | 49.54 | 53.35 |
| **YOLOv8n** | 0.34 | 0.64 | 0.92 | 1.23 | 1.25 | 0.65 | 0.89 | 1.03 |
| **YOLOv8s** | 0.36 | 0.39 | 0.80 | 1.07 | 1.08 | 0.60 | 0.86 | 0.95 |
| **YOLOv8m** | 0.34 | 0.43 | 0.52 | 0.90 | 1.00 | 0.58 | 0.72 | 0.87 |
| **YOLOv8l** | 0.35 | 0.48 | 0.65 | 0.70 | 0.88 | 0.49 | 0.62 | 0.75 |
| **YOLOv8x** | 0.31 | 0.45 | 0.61 | 0.66 | 0.72 | 0.41 | 0.58 | 0.67 |
| **Faster R-CNN** | 5.13 | 9.04 | 16.02 | 18.51 | 19.75 | 0.09 | 1.42 | 17.23 |
| **RetinaNet** | 8.84 | 13.94 | 21.47 | 23.89 | 25.57 | 1.97 | 0.12 | 21.97 |
| **Swin-T** | 2.99 | 6.06 | 12.39 | 15.30 | 18.18 | 12.18 | 17.38 | 0.18 |

</center>

<p align="center">
    Table for cross-model transferability among commonly used detection models (in mAP) of various-sized YOLO's, Faster R-CNN, RetinaNet, and Swin Transformer, at confidence thresholds of 0.50, validating on <b>PASCAL VOC 2012 validation set</b>.
</p>


#### Quantitative Results in Success Attack Rate

<center>

| Added Perturbation| YOLOv8n | YOLOv8s | YOLOv8m | YOLOv8l | YOLOv8x | Faster R-CNN | RetinaNet | Swin-T | Avg | 
| :---------------: | :------: | :----: | :------: | :----: |:------: | :----: | :------: | :----: | :----: |
| **YOLOv8n** | 99.76 | 45.52 | 31.88 | 27.45 | 26.24 | 51.36 | 53.32 | 46.97 | 47.81 | 
| **YOLOv8s** | 86.74 | 99.91 | 54.81 | 46.89 | 43.9 | 65.30 | 68.08 | 57.93 | 65.44 |
| **YOLOv8m** | 91.17 | 86.92 | 99.95 | 66.31 | 61.72 | 74.80 | 78.12 | 67.08 | 78.26 |
| **YOLOv8l** | 93.25 | 89.42 | 81.34 | 99.95 | 71.59 | 77.20 | 81.00 | 68.65 | 82.80 |
| **YOLOv8x** | 94.33 | 91.19 | 85.21 | 83.31 | 99.88 | 80.93 | 84.37 | 73.61 | 86.60 |
| **Faster R-CNN** | 84.58 | 79.07 | 71.58 | 66.38 | 65.12 | 99.64 | 97.38 | 60.89 | 78.08 |
| **RetinaNet** | 76.00 | 69.96 | 61.55 | 56.32 | 55.00 | 92.47 | 98.69 | 50.72 | 70.09 |
| **Swin-T** | 88.10 | 82.47 | 74.34 | 68.10 | 63.77 | 59.50 | 62.01 | 99.69 | 74.75 |

</center>

<p align="center">
    Table for cross-model transferability among commonly used detection models (in percentage) of various-sized YOLO's, Faster R-CNN, RetinaNet, and Swin Transformer, at confidence thresholds of 0.50, validating on <b>MS COCO 2017 validation set</b>.
</p>

<center>

| Added Perturbation| YOLOv8n | YOLOv8s | YOLOv8m | YOLOv8l | YOLOv8x | Faster R-CNN | RetinaNet | Swin-T | Avg | 
| :---------------: | :------: | :----: | :------: | :----: |:------: | :----: | :------: | :----: | :----: |
| **YOLOv8n** | 99.25 | 98.82 | 98.49 | 98.06 | 98.05 | 98.59 | 98.20 | 98.07 | 98.44 |
| **YOLOv8s** | 99.20 | 99.28 | 98.68 | 98.31 | 98.31 | 98.70 | 98.26 | 98.22 | 98.62 |
| **YOLOv8m** | 99.25 | 99.21 | 99.14 | 98.58 | 98.44 | 98.74 | 98.55 | 98.37 | 98.78 |
| **YOLOv8l** | 99.22 | 99.12 | 98.93 | 98.90 | 98.63 | 98.94 | 98.75 | 98.59 | 98.89 |
| **YOLOv8x** | 99.31 | 99.17 | 99.00 | 98.96 | 98.88 | 99.11 | 98.83 | 98.74 | 99.00 |
| **Faster R-CNN** | 88.64 | 83.40 | 73.65 | 70.84 | 69.14 | 99.80 | 97.13 | 67.70 | 81.29 |
| **RetinaNet** | 80.42 | 74.40 | 64.69 | 62.36 | 60.05 | 95.73 | 99.76 | 58.82 | 74.53 |
| **Swin-T** | 93.38 | 88.87 | 79.62 | 75.89 | 71.59 | 73.60 | 64.92 | 99.66 | 80.94 |

</center>

<p align="center">
    Table for cross-model transferability among commonly used detection models (in percentage) of various-sized YOLO's, Faster R-CNN, RetinaNet, and Swin Transformer, at confidence thresholds of 0.50, validating on <b>PASCAL VOC 2012 validation set</b>.
</p>


## Ultralytics YOLOv8 Attack

* The requirements are same as: [YOLOv8](https://github.com/ultralytics/ultralytics) and [YOLO-V8-CAM](https://github.com/rigvedrs/YOLO-V8-CAM).


### Files structure
Files should be placed as the following folder structure:

```
root
├── data
│   ├── coco
│   │   │── images
│   │   │   │── train
|   |   |   |   |── 000000109622.jpg
|   |   |   |   |── ....
│   │   │   │── val
|   |   |   |   |── 000000000139.txt
|   |   |   |   |── ....
│   │   │── labels
│   │   │   │── train
|   |   |   |   |── 000000109622.jpg
|   |   |   |   |── ....
│   │   │   │── val
|   |   |   |   |── 000000000139.txt
|   |   |   |   |── ....
├── ultralytics
├── yolo_cam
|   |── activations_and_gradients
|   |── ...
├── video_demo
|   |── broadway.mp4
|   |── ...
├── utils
|   ├── CAM_utils.py
|   ├── tools
├── yolov8_coco_attack.py
├── yolov8_voc_attack.py
├── class_activation_map.py
├── evaluation
|   ├── statistics.py
|   |── ....
├── server.py
├── client_capture.py
├── client_realsense.py
├── video_attack.py
```
### Usage
1. Run attack on MS COCO 2017 dataset:
    ```
    python yolov8_coco_attack.py
    ```

2. Run attack on PASCAL VOC 2012 dataset:
    ```
    python yolov8_voc_attack.py
    ```

3. Start server with object detector:

    ```
    python server.py
    ```

4. Stream images from the captured device:

    ```
    python client_capture.py
    ```

5. Run attacks on video:
    ```
    python video_attack.py
    ```



6. Visualize class activation map before and after attack:
    ```
    python class_activation_map.py
            --img_folder {path/to/img/before/and/after/attack} # contain only 2 imgs (before and after)
            --model_name {n/s/m/l/x}
            --det_threshold {0.5}
            --cam_threshold {0.4} # must be less than det_threshold
    ```

<!-- [MMDetection](https://github.com/open-mmlab/mmdetection) Model Zoo (Faster_RCNN, RetinaNet, Swin) -->
## MMDetection Attack

* Data Preparation: Set up the data the same as [MMDetection](https://github.com/open-mmlab/mmdetection).

* Pretrained Models: Download [pre-trained models](https://github.com/open-mmlab/mmdetection) from MMDetection Model Zoo and put them in `./checkpoints`.

### Inference Demo Images

```
python demo/image_demo.py \
       path/to/image/for/visualize \
       path/to/custom/config/saved/in/folder/custom_configs \
       checkpoint/for/each/model/in/folder/checkpoints
```
This script will save the image that needs to be inference in the `outputs` folder. Custom config for `COCO` dataset saved in `custom_configs/coco`, and for `VOC2012` saved in `custom_configs/voc`.


### Usage
1. Run attack on MS COCO 2017 dataset:
    ```
    python mmdet_coco_attack.py
    ```

2. Run attack on PASCAL VOC 2012 dataset:
    ```
    python mmdet_voc_attack.py
    ```

<!-- CITING -->

## Citing

```

```
