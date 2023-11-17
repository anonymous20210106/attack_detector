import gc
import os
import shutil
from glob import glob

import cv2
import matplotlib.pyplot as plt
import torch

from ultralytics import YOLO
from ultralytics.yolo.cfg import cfg2dict, get_cfg
from ultralytics.yolo.data.build import build_dataloader
from ultralytics.yolo.v8.detect.train import DetectionTrainer
from utils.tools import (boxes_to_coor, compute_normalized_cross_correlation,
                         cut_images, delete_temp_img, save_visualize_image, save_loss, img2vidlst)

from moviepy.editor import *
 

def video_to_image(name_video, total_frame):
    cam = cv2.VideoCapture(f"./video_demo/{name_video}.MOV")
    try:
        
        # creating a folder named data
        if not os.path.exists(f'{name_video}'):
            os.makedirs(f'{name_video}')
            
        if not os.path.exists(f'{name_video}/images'):
            os.makedirs(f'{name_video}/images')
        
        if not os.path.exists(f'{name_video}/images/val'):
            os.makedirs(f'{name_video}/images/val')
            
        if not os.path.exists(f'{name_video}/labels'):
            os.makedirs(f'{name_video}/labels')
            
        if not os.path.exists(f'{name_video}/labels/val'):
            os.makedirs(f'{name_video}/labels/val')
        
    # if not created then raise error
    except OSError:
        print ('Error: Creating directory of data')
        # frame
    
    
    currentframe = 0

    while(True):
    
        # reading from frame
        ret,frame = cam.read()
        
        if currentframe == total_frame:
            break
        if ret:
            # if video is still left continue creating images
            name = f'{name_video}/images/val/' + str(currentframe) + '.jpg'
            print ('Creating...' + name)
            resized = cv2.resize(frame, (640,480))
            # writing the extracted images
            cv2.imwrite(name, resized)
    
            # increasing counter so that it will
            # show how many frames are created
            currentframe += 1
        else:
            break
    
    # Release all space and windows once done
    cam.release()
    cv2.destroyAllWindows()

def create_labels(name_video):
    model = YOLO("yolov8x.pt")
    model.conf = 0.55
    path_img = glob(f'{name_video}/images/val/*')
    
    for path in path_img:
        predicts = model.predict(path)[0].boxes
        name_labels = int(os.path.split(path)[1][:-4])
        with open(f'./{name_video}/labels/val/{name_labels}.txt', 'w') as f:
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
        print(name_labels)
    torch.cuda.empty_cache()
    gc.collect()
    return


def make_video(name_video, model_name, conf):
    path = './cloud'
    isExist = os.path.exists(path)
    if isExist:
        shutil.rmtree('./cloud')
    os.mkdir('./cloud')
    model = YOLO(f"yolov8{model_name}.pt")
    model.conf = conf
    path_attacked = sorted(glob(f"./img_results/{name_video}_yolov8{model_name}.pt_{conf}/images_full/val/*_*_*"), key= lambda x: int(os.path.split(x)[1].split("_")[0]))

    
    for path in path_attacked:
        image_base = os.path.join(os.path.split(path)[0],os.path.split(path)[1].split('_')[0] + '_base.jpg')
     
        if (float(os.path.split(path)[1].split('_')[3][:-4]) < 0.9):
            continue
        plt.imsave(f'./cloud/{os.path.split(image_base)[1]}',cv2.cvtColor(model.predict(image_base)[0].plot(), cv2.COLOR_BGR2RGB))
        plt.imsave(f'./cloud/{os.path.split(path)[1]}',cv2.cvtColor(model.predict(path)[0].plot(), cv2.COLOR_BGR2RGB))
    
    path_bb_base =  sorted(glob("./cloud/*_base*"), key= lambda x: int(os.path.split(x)[1].split("_")[0]))
    path_bb_attack =  sorted(glob("./cloud/*_*_*"), key= lambda x: int(os.path.split(x)[1].split("_")[0]))
   
    img2vidlst(f'{name_video}_base.mp4', path_bb_base, fps=24)
    img2vidlst(f'{name_video}_attacked.mp4', path_bb_attack, fps=24)
    clip1 = VideoFileClip(f'{name_video}_base.mp4')
    clip2 = VideoFileClip(f'{name_video}_attacked.mp4')
    clips = [[clip1, clip2]]
    final = clips_array(clips)
    final.ipython_display(width = 480)


def delete_tmp(folder_name):
    isExist = os.path.exists(f'{folder_name}_attacked.mp4')
    
    if isExist == True:
        os.remove(f'{folder_name}_attacked.mp4')
   
    isExist = os.path.exists(f'{folder_name}_base.mp4')
    if isExist == True:
        os.remove(f'{folder_name}_base.mp4')
    isExist = os.path.exists('cloud')
    if isExist == True:
        shutil.rmtree('cloud')
    isExist = os.path.exists(f'{folder_name}')
    if isExist == True:
        shutil.rmtree(f'{folder_name}')


def video_attack(successful_rate, folder_name, name_yolo, conf, total_frame):

    video_to_image(folder_name, total_frame)
    create_labels(folder_name)
    
    img_path = f'./{folder_name}/images/val'
    
    data_info_path = f'./ultralytics/datasets/{folder_name}.yaml'

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
        
        model = YOLO(f'yolov8{name_yolo}.pt')
        model.conf = conf
        
        image_to_attack_ori = batch_tensor[:,:,y1:y2,x1:x2].detach().clone().cuda()
        image_to_attack = batch_tensor[:,:,y1:y2,x1:x2].cuda()
        attack_grad     = alpha[:,:,y1:y2,x1:x2].cuda()

        step = 10
        loss_log_image.append(loss)
       
        res_ori = model.predict(path_img)
        number_full_objects = None
        folder_result = folder_name +"_"+cfg.get('model')+"_"+ str(model.conf) 
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
                step = 10
                
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

    make_video(folder_name,name_yolo,conf)
    delete_tmp(folder_name)



if __name__ == '__main__':
    folder_name = "broadway"
    successful_rate = 1
    name_yolo = "n"
    conf = 0.5
    total_frame = 2000
    # make_video(folder_name,name_yolo,conf)
    video_attack(successful_rate, folder_name, name_yolo, conf, total_frame)
   