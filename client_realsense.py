import gc
import os
import time
#%matplotlib inline
from glob import glob

import cv2
# Ultralytics YOLO ??, AGPL-3.0 license
import numpy as np
import pyrealsense2 as rs
import torch
import socket

from ultralytics import YOLO
from ultralytics.yolo.cfg import cfg2dict, get_cfg
from ultralytics.yolo.data.build import build_dataloader
from ultralytics.yolo.v8.detect.train import DetectionTrainer
from utils.tools import (boxes_to_coor, cut_images, delete_temp_img,
                         prepare_cloud_folder, save_visualize_image)

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)



def create_labels(path):
   
    model = YOLO("yolov8m.pt")
    model.conf = 0.55
    predicts = model.predict(path)[0].boxes
    with open(f'./camera_attack/labels/val/attack_frame.txt', 'w') as f:
        
        for i in range(predicts.cls.shape[0]):
         
            bbox_str = ""
            for j in range(4):
                bbox_str += f"{float(predicts.xywhn[i][j])}"
                if j < 3:
                    bbox_str += " "
            f.write(str(int(predicts.cls[i])))
            f.write(" ")
            f.write(bbox_str)
            f.write('\n')
    torch.cuda.empty_cache()
    gc.collect()


config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)


#Skip some first frames
for _ in range(30):
    frames = pipeline.wait_for_frames()


model = YOLO("yolov8n.pt")
model.conf = 0.55
yaml_path       = './ultralytics/yolo/cfg/my_config.yaml'
trainer = DetectionTrainer(yaml_path)
trainer.setup_model()
trainer.set_model_attributes()
trainer.model.train()
trainer.model.cuda()

try:
    
    # Create a TCP/IP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Connect the socket to the port where the server is listening
    server_address = ('192.168.0.101', 7777)
    print( 'connecting to %s port %s' % server_address)
    sock.connect(server_address)
    while True:
        
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
       
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            continue

        
        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Brightness control
        darkness = 90
        color_image = np.where(color_image > darkness, color_image - darkness, 0)
        
       
        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape

        start = time.time()
         
        prepare_cloud_folder()
        img_path            = f'./camera_attack/images/val'
        color_image         = cv2.resize(color_image, (640,480))
        color_colormap_dim  = cv2.imwrite(f'{img_path}/attack_frame.jpg',color_image)
        img_result = color_image
        create_labels(f'{img_path}/attack_frame.jpg')
        if model.predict(color_image)[0].boxes.shape[0] > 0:
    
            data_info_path  = './ultralytics/datasets/coco2017.yaml'

            data_info       = cfg2dict(data_info_path)
            
            cfg             = get_cfg(yaml_path)

            #data loader
            loader = build_dataloader(cfg= cfg, batch=1, img_path=img_path,data_info=data_info)
            
            for i, batch in enumerate(loader[0]):
                path_img        = loader[1].im_files[i]
                name_img        = os.path.split(path_img)[-1]
            
                
                    
                batch_tensor    = torch.Tensor(batch['img'].float()).cuda()
                
                img_before      = cv2.imread(path_img)
                
                x,y,_           = img_before.shape

                x_new,y_new     = batch_tensor.shape[-2:]

                x1,y1,x2,y2     = cut_images((x,y,3),[x_new,y_new])
                
                

                batch_tensor.requires_grad_()
                batch_tensor.retain_grad()
                b = trainer.model(batch_tensor)

                loss, loss_each_module = trainer.criterion(b, batch)
                

                loss.retain_grad()
                loss.backward()

                alpha = batch_tensor.grad
                
    
                
                image_to_attack_ori = batch_tensor[:,:,y1:y2,x1:x2].detach().clone().cuda()
                image_to_attack = batch_tensor[:,:,y1:y2,x1:x2].cuda()
                attack_grad     = alpha[:,:,y1:y2,x1:x2].cuda()

                step = 10000
        
            
                res_ori = model.predict(path_img)
                number_full_objects = None
                folder_result = 'cloud_attack_realsens'
                for number_of_steps in range(60000):
                    res = model.predict(path_img)
                    number_object = res[0].boxes.xyxy.shape[0]
                    
                    print(number_object)
                    print(res[0].boxes.conf)
                    
                    
                    if number_of_steps == 0:
                        save_visualize_image(image_to_attack/255, folder_result ,name_img, 'base')
                        path_img           =   f'./img_results/{folder_result}/images_full/val/{name_img[:-4]}_base.jpg'
                        step = 0
                        res_ori = model.predict(path_img)
                        number_full_objects = res_ori[0].boxes.xyxy.shape[0]
                        new_image           = image_to_attack
                        continue
                        
                    else:
                        step = 10000
                        
                    
                    
                    if number_object <= 0.1*number_full_objects or number_of_steps == 60000 - 1:
                        ncc = 0
                        if number_full_objects != 0:
                            for index in range(number_full_objects):
                            
                                x1_bbox,y1_bbox,x2_bbox,y2_bbox                    = boxes_to_coor(res_ori[0].boxes.xyxy[index])
                                ########################################
                                image1 = image_to_attack[:,:,y1_bbox:y2_bbox,x1_bbox:x2_bbox] 
                                image2 = image_to_attack_ori[:,:,y1_bbox:y2_bbox,x1_bbox:x2_bbox]
                              
                              
                        save_visualize_image(new_image/255, folder_result ,name_img, f'ncc_mean_')
                        break
                        
                
                    masks_object_attack                                            = torch.zeros_like(attack_grad)
                    for index in range(number_object):
                        
                        x1_bbox,y1_bbox,x2_bbox,y2_bbox                            = boxes_to_coor(res[0].boxes.xyxy[index])
                        masks_object_attack[:,:,y1_bbox:y2_bbox,x1_bbox:x2_bbox]  += 1
                        
                    
                    masks_object_attack                    = torch.where(masks_object_attack > 0, 1, 0)
                    new_image                              = image_to_attack + step*attack_grad*masks_object_attack
                    image_to_attack                        = new_image  
                    
                        
            
                    save_visualize_image(new_image/255, folder_result ,name_img, f'{number_object}_{number_of_steps}')
                    
                    path_img           =   f'./img_results/{folder_result}/images_full/val/{name_img[:-4]}_{number_object}_{number_of_steps}.jpg'
                    
                    with torch.no_grad():
                        batch_tensor[:,:,y1:y2,x1:x2] = image_to_attack
                        
                    batch_tensor.requires_grad_()
                    batch_tensor.retain_grad()
                    b = trainer.model(batch_tensor)

                    loss, loss_each_module = trainer.criterion(b, batch)
                    

                    loss.retain_grad()
                    loss.backward()
                    alpha = batch_tensor.grad
                    image_to_attack = batch_tensor[:,:,y1:y2,x1:x2].cuda()
                    attack_grad     = alpha[:,:,y1:y2,x1:x2].cuda()
                
                
                    torch.cuda.empty_cache()
                    gc.collect()
            
                
                delete_temp_img(folder_result)
        
                result_path = glob(f'img_results/cloud_attack_realsens/images_full/val/*_ncc_mean_*')
                img_result = cv2.imread(result_path[0])

        #send data
        data = bytes(img_result)
        l = len(data)
        print('l = ', type(l), '>',  len(l.to_bytes(4, 'little')))
        # print('len = ', len(int.to_bytes(l)))
        sock.sendall(l.to_bytes(4, 'little'))
        sock.sendall(data) #resize frame, then convert it in to bytes
        # time.sleep(0.5)
        # i += 1
        # if i > 2:
        #     break
        cv2.destroyAllWindows()
        
        if (cv2.waitKey(1) and 0xFF == ord('q')):
                    break
        
            
        end = time.time()
        print("\n ================= \n")
        print(end - start)
        print("\n ================= \n")
        
        # If depth and color resolutions are different, resize color image to match depth image for display
        if depth_colormap_dim != color_colormap_dim:
            resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
            images = np.hstack((resized_color_image, depth_colormap))
        else:
            images = np.hstack((color_image, depth_colormap))





        

        # Show images
        #cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        # cv2.imshow('RealSense', images)
        #cv2.imshow('RealSense', img_result)

        #cv2.waitKey(1)

finally:

    # Stop streaming
    pipeline.stop()

print('closing socket')
sock.close()