from utils.tools import grad_attack_problem_solving,show_imgs_tensor
import cv2
import torch
from ultralytics.yolo.v8.detect.train import DetectionTrainer
from ultralytics.yolo.v8.segment.train import SegmentationTrainer
from ultralytics.yolo.data.dataloaders.v5loader import create_dataloader
from ultralytics.yolo.v8.segment.train import Loss
from ultralytics import YOLO
from torch.autograd import Variable
import torch
import os
import json
from pathlib import Path
import torch
from tqdm import tqdm
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.yolo.cfg import get_cfg
from ultralytics.yolo.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.yolo.utils import DEFAULT_CFG, LOGGER, RANK, SETTINGS, TQDM_BAR_FORMAT, callbacks, colorstr, emojis
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.utils.files import increment_path
from ultralytics.yolo.utils.ops import Profile
from ultralytics.yolo.utils.torch_utils import de_parallel, select_device, smart_inference_mode
# Ultralytics YOLO ??, AGPL-3.0 license
from pathlib import Path
import numpy as np
from ultralytics.yolo.engine.validator import BaseValidator
from ultralytics.yolo.utils import DEFAULT_CFG, LOGGER, colorstr, ops
from ultralytics.yolo.utils.checks import check_requirements
from ultralytics.yolo.utils.metrics import ConfusionMatrix, DetMetrics, box_iou
from ultralytics.yolo.utils.plotting import output_to_target, plot_images
from ultralytics.yolo.utils.torch_utils import de_parallel
#from tools import calculate_loss_of_tensor, show_imgs_tensor,save_visualize_loss,cut_images,save_visualize_loss_segment,scale_masks,show_imgs_3_channel_tensor,set_state,copy_images_to_folder,show_imgs_tensor, plot_loss_log, grad_attack_each_iter,save_adversarial, setup_folder,check_gray,grad_attack_iter, plot_bbox, plot_bbox_border, reduce_border
from ultralytics.yolo.utils.instance import Bboxes, Instances 
from utils.tools import  delete_temp_img, save_loss,boxes_to_coor,show_imgs_tensor,save_visualize_image,scale_masks,cut_images,set_state,copy_images_to_folder,show_imgs_tensor, plot_loss_log, grad_attack_each_iter,save_adversarial, setup_folder,check_gray,grad_attack_iter, plot_bbox, plot_bbox_border, reduce_border
from utils.tools import compute_normalized_cross_correlation, delete_min_object_iter
import cv2
import matplotlib.pyplot as plt
#%matplotlib inline
from ultralytics.yolo.utils.ops import scale_boxes
import glob
from ultralytics.yolo.data.build import build_dataloader
from ultralytics.yolo.cfg import cfg2dict
from ultralytics.yolo.cfg import get_cfg
import gc

def yolov8_coco_attack_ncc(ncc_value, folder_name, conf, step_size, iter_start, iter_end):
    img_path = './datasets/coco2017/images/val'
    
    data_info_path = './ultralytics/datasets/coco2017.yaml'

    yaml_path = './ultralytics/yolo/cfg/my_config.yaml'
    
    data_info = cfg2dict(data_info_path)
    cfg = get_cfg(yaml_path)

    #data loader
    loader = build_dataloader(cfg= cfg, batch=1, img_path=img_path,data_info=data_info)

    #trainer
    trainer = DetectionTrainer(yaml_path)

    trainer.setup_model()
    trainer.set_model_attributes()
    trainer.model.train()
    trainer.model.cuda()


    for i, batch in enumerate(loader[0]):
        if (i < iter_start):
            continue
        if (i > iter_end):
            break
        loss_log_image  = []
        path_img        = loader[1].im_files[i]
        name_img        = os.path.split(path_img)[-1]

        batch_tensor    = torch.Tensor(batch['img'].float()).cuda()
        
        img_before      = cv2.imread(path_img)
        
        x,y,_           = img_before.shape

        x_new,y_new     = batch_tensor.shape[-2:]

        x1,y1,x2,y2     = cut_images((x,y,3),[x_new,y_new])
        
        name_fake_img = os.path.split(name_img)[-1]

        batch_tensor.requires_grad_()
        batch_tensor.retain_grad()
        b = trainer.model(batch_tensor)

        loss, loss_each_module = trainer.criterion(b, batch)
        

        loss.retain_grad()
        loss.backward()

        alpha = batch_tensor.grad
        
        model = YOLO(cfg.get('model'))
        model.conf = conf
        
        
        image_to_attack_ori = batch_tensor[:,:,y1:y2,x1:x2].detach().clone().cuda()
        image_to_attack = batch_tensor[:,:,y1:y2,x1:x2].cuda()
        attack_grad     = alpha[:,:,y1:y2,x1:x2].cuda()

        step = 0
        loss_log_image.append(loss)
        bbox_list = []
        folder_result = folder_name +"_"+cfg.get('model')+"_"+ str(ncc_value) +'_ncc_' + str(conf)+"_" + str(iter_start) + "-"+ str(iter_end)
        num_object_min = 0
        num_iter_min   = 0
        for number_of_steps in range(5000):
            
            res = model.predict(path_img)
            number_object = res[0].boxes.xyxy.shape[0]
            
            print(number_object)
            print(res[0].boxes.conf)
            
           
            if number_of_steps == 0:
                
                save_visualize_image(image_to_attack/255, folder_result ,name_img, 'base')
                path_img           =   f'./img_results/{folder_result}/images_full/val/{name_img[:-4]}_base.jpg'
                step = 0
                num_object_min += number_object
                num_iter_min   += number_of_steps
                save_visualize_image(image_to_attack/255, folder_result ,name_img, f'{number_object}_{number_of_steps}')
                continue
                
            else:
                step = step_size
            
            if number_object < num_object_min:
                num_object_min = 0
                num_iter_min   = 0
                num_object_min += number_object
                num_iter_min   += number_of_steps

            
            count = 0
            masks_object_attack                                            = torch.zeros_like(attack_grad)
            for index in range(number_object):
                x1_bbox,y1_bbox,x2_bbox,y2_bbox                            = boxes_to_coor(res[0].boxes.xyxy[index])
                ########################################
                image1 = image_to_attack[:,:,y1_bbox:y2_bbox,x1_bbox:x2_bbox] 
                image2 = image_to_attack_ori[:,:,y1_bbox:y2_bbox,x1_bbox:x2_bbox]
                ncc    = compute_normalized_cross_correlation(image1, image2)
                print(ncc)
                if ncc < ncc_value:
                    count += 1
                    continue        
                ########################################
                masks_object_attack[:,:,y1_bbox:y2_bbox,x1_bbox:x2_bbox]  += 1
                
            
            masks_object_attack                    = torch.where(masks_object_attack > 0, 1, 0)
            new_image                              = image_to_attack + step*attack_grad*masks_object_attack
            image_to_attack                        = new_image  
            
                
                    
            save_visualize_image(new_image/255, folder_result ,name_img, f'{number_object}_{number_of_steps}')
            if count == number_object:
                break
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
        
            loss_log_image.append(loss)
            
            torch.cuda.empty_cache()
            gc.collect()
        delete_min_object_iter(folder_result, name_img, num_object_min, num_iter_min)
        save_loss(loss_log_image, folder_result, name_img)
        
        
def yolov8_coco_successful_rate_attack(successful_rate, folder_name, conf, iter_start, iter_end):
    img_path = './datasets/coco2017/images/val'
    
    data_info_path = './ultralytics/datasets/coco2017.yaml'

    yaml_path = './ultralytics/yolo/cfg/my_config.yaml'
    
    data_info = cfg2dict(data_info_path)
    cfg = get_cfg(yaml_path)

    #data loader
    loader = build_dataloader(cfg= cfg, batch=1, img_path=img_path,data_info=data_info)

    #trainer
    trainer = DetectionTrainer(yaml_path)

    trainer.setup_model()
    trainer.set_model_attributes()
    trainer.model.train()
    trainer.model.cuda()
    
    for i, batch in enumerate(loader[0]):
        
        if i < iter_start:
            continue
        if i > iter_end:
            break
        
        loss_log_image  = []
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

        loss, _ = trainer.criterion(b, batch)

        loss.retain_grad()
        loss.backward()

        alpha = batch_tensor.grad
        
        model = YOLO(cfg.get('model'))
        model.conf = conf
        
        image_to_attack_ori = batch_tensor[:,:,y1:y2,x1:x2].detach().clone().cuda()
        image_to_attack = batch_tensor[:,:,y1:y2,x1:x2].cuda()
        attack_grad     = alpha[:,:,y1:y2,x1:x2].cuda()

        step = 100
        loss_log_image.append(loss)
       
        res_ori = model.predict(path_img)
        number_full_objects = None
        folder_result = folder_name +"_"+cfg.get('model')+"_"+ str(model.conf) +'_numobject_' + str(iter_start) + "_" + str(iter_end)
        for number_of_steps in range(5000):
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
                step = 100
            
            if number_object <=(1-successful_rate)*number_full_objects or number_of_steps == 60000 - 1:
                ncc = 0
                if number_full_objects != 0:
                    for index in range(number_full_objects):
                    
                        x1_bbox,y1_bbox,x2_bbox,y2_bbox                    = boxes_to_coor(res_ori[0].boxes.xyxy[index])
                        ########################################
                        print(image_to_attack.shape)
                        image1 = image_to_attack[:,:,y1_bbox:y2_bbox,x1_bbox:x2_bbox] 
                        print(image1.shape)
                        image2 = image_to_attack_ori[:,:,y1_bbox:y2_bbox,x1_bbox:x2_bbox]
                        print(image2.shape)
                        ncc    += compute_normalized_cross_correlation(image1, image2)
                    ncc /= number_full_objects
                else:
                    ncc += 1
                save_visualize_image(new_image/255, folder_result ,name_img, f'ncc_mean_{ncc}')
                break
                
           
            masks_object_attack                                            = torch.zeros_like(attack_grad)
            for index in range(number_object):
                
                x1_bbox,y1_bbox,x2_bbox,y2_bbox                            = boxes_to_coor(res[0].boxes.xyxy[index])
                masks_object_attack[:,:,y1_bbox:y2_bbox,x1_bbox:x2_bbox]  += 1
                
            masks_object_attack                    = torch.where(masks_object_attack > 0, 1, 0)
            new_image                              = image_to_attack + step*attack_grad*masks_object_attack
            image_to_attack                        = new_image  
            save_visualize_image(new_image/255, folder_result ,name_img, f'{number_object}_{number_of_steps}')
            save_visualize_image((image_to_attack - image_to_attack_ori), folder_result ,name_img, f'{number_object}_{number_of_steps}_diff')
         
            path_img           =   f'./img_results/{folder_result}/images_full/val/{name_img[:-4]}_{number_object}_{number_of_steps}.jpg'
            
            with torch.no_grad():
                batch_tensor[:,:,y1:y2,x1:x2] = image_to_attack
                
            batch_tensor.requires_grad_()
            batch_tensor.retain_grad()
            b = trainer.model(batch_tensor)

            loss, _ = trainer.criterion(b, batch)
            loss.retain_grad()
            loss.backward()
            alpha = batch_tensor.grad
            image_to_attack = batch_tensor[:,:,y1:y2,x1:x2].cuda()
            attack_grad     = alpha[:,:,y1:y2,x1:x2].cuda()
        
            loss_log_image.append(loss)
            
            torch.cuda.empty_cache()
            gc.collect()
        save_loss(loss_log_image, folder_result, name_img)        
        
        delete_temp_img(folder_result)
        
        
if __name__ == '__main__':


    folder_name = 'coco2017'
    ncc_threshold = 0.6
    conf_threshold = 0.5
    successful_rate = 1
    
    select = 99
    if select == 1:
        iter_start  = 0
        iter_end    = 2500
        
    elif select == 99:
        iter_start  = 4995
        iter_end    = 5001
    else:
        iter_start = 0
        iter_end   = 5000

    yolov8_coco_attack_ncc(ncc_threshold, folder_name, conf_threshold, iter_start, iter_end)
    
    yolov8_coco_successful_rate_attack(successful_rate, folder_name, conf_threshold, iter_start, iter_end)