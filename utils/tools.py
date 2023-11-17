from torchvision.utils import save_image
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
import os
from ultralytics.yolo.utils.instance import  Instances 
import cv2
import matplotlib.pyplot as plt
import numpy as np 
import shutil
import yaml
from ultralytics.yolo.v8.detect.train import DetectionTrainer
from ultralytics import YOLO
from ultralytics.yolo.data.dataloaders.v5loader import create_dataloader
from ultralytics.yolo.v8.segment.train import SegmentationTrainer
from ultralytics.yolo.v8.segment.train import Loss
import pandas as pd
from glob import glob

def img2vidlst(output_video_path,lst_images, fps=1):
    
    first_image_path = lst_images[0]
    first_image = cv2.imread(first_image_path)
    height, width, _ = first_image.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    for image_path in lst_images:
        #img_path = os.path.join(images_folder, image_path)
        #image_path = os.path.join(images_folder, image_file)
        image = cv2.imread(image_path)
        video_writer.write(image)

    video_writer.release()
    cv2.destroyAllWindows()

def save_loss(loss_log_image, name_of_fol, name_image):
    path = f'./img_results/{name_of_fol}'
    isExist = os.path.exists(path)
    if isExist == False:
        os.mkdir(path)
    path = f'./img_results/{name_of_fol}/loss'
    isExist = os.path.exists(path)
    if isExist == False:
        os.mkdir(path)
    
    #np.savetxt(f'{name_image}_loss.txt',torch.tensor(loss_log_image).detach().cpu().numpy(), fmt = '%.18f')
    plt.plot(torch.tensor(loss_log_image).detach().cpu().numpy())
    plt.savefig(f'./img_results/{name_of_fol}/loss/{name_image}')
    plt.clf()
    plt.close()
    
    

def save_ncc(loss_log_image, name_of_fol, name_image):
    path = f'./results/{name_of_fol}'
    isExist = os.path.exists(path)
    if isExist == False:
        os.mkdir(path)
    path = f'./results/{name_of_fol}/ncc'
    isExist = os.path.exists(path)
    if isExist == False:
        os.mkdir(path)
    
    np.savetxt(f'{name_image}_ncc.txt',torch.tensor(loss_log_image).detach().cpu().numpy(), fmt = '%.18f')
    plt.plot(torch.tensor(loss_log_image).detach().cpu().numpy())
    plt.savefig(f'./results/{name_of_fol}/ncc/{name_image}')
    plt.clf()
    plt.close()
    

def boxes_to_coor(box):
    l = int(round(float(box[0])))
    t = int(round(float(box[1])))
    r = int(round(float(box[2])))
    b = int(round(float(box[3])))
    return l, t, r, b

def reduce_border(batch_tensor,shape):
    x,y = shape
    offsetx = int((640 - x) / 2)
    offsety = int((640 - y) / 2)
    batch_tensor = batch_tensor[:,:,offsety:y + offsety,offsetx:x+offsetx]
    return batch_tensor
    

def plot_bbox_border(image_path, label_path,shape):
    
    x,y = shape
    offsetx = int((640 - x) / 2)
    offsety = int((640 - y) / 2)
    if offsetx > offsety:
        img = cv2.imread(image_path)[:,offsetx:x+offsetx,:]
    
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        img = cv2.imread(image_path)[offsety:y + offsety,:,:]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
 
    dh, dw, _ = img.shape

    fl = open(label_path, 'r')
    data = fl.readlines()
    fl.close()

    for dt in data:

        # Split string to float
        _, x, y, w, h = map(float, dt.split(' '))

        # Taken from https://github.com/pjreddie/darknet/blob/810d7f797bdb2f021dbe65d2524c2ff6b8ab5c8b/src/image.c#L283-L291
        # via https://stackoverflow.com/questions/44544471/how-to-get-the-coordinates-of-the-bounding-box-in-yolo-object-detection#comment102178409_44592380
        l = int((x - w / 2) * dw)
        r = int((x + w / 2) * dw)
        t = int((y - h / 2) * dh)
        b = int((y + h / 2) * dh)
        
        if l < 0:
            l = 0
        if r > dw - 1:
            r = dw - 1
        if t < 0:
            t = 0
        if b > dh - 1:
            b = dh - 1

        cv2.rectangle(img, (l, t), (r, b), (0, 0, 255), 2)

    plt.imshow(img)
    plt.show()
    
def plot_bbox(image_path, label_path):

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    dh, dw, _ = img.shape

    fl = open(label_path, 'r')
    data = fl.readlines()
    fl.close()

    for dt in data:

        # Split string to float
        _, x, y, w, h = map(float, dt.split(' '))

        # Taken from https://github.com/pjreddie/darknet/blob/810d7f797bdb2f021dbe65d2524c2ff6b8ab5c8b/src/image.c#L283-L291
        # via https://stackoverflow.com/questions/44544471/how-to-get-the-coordinates-of-the-bounding-box-in-yolo-object-detection#comment102178409_44592380
        l = int((x - w / 2) * dw)
        r = int((x + w / 2) * dw)
        t = int((y - h / 2) * dh)
        b = int((y + h / 2) * dh)
        
        if l < 0:
            l = 0
        if r > dw - 1:
            r = dw - 1
        if t < 0:
            t = 0
        if b > dh - 1:
            b = dh - 1

        cv2.rectangle(img, (l, t), (r, b), (0, 0, 255), 2)

    plt.imshow(img)
    plt.show()



def save_adversarial(img, name_of_fol ,name):
    path = f'./datasets/{name_of_fol}'
    isExist = os.path.exists(path)
    if isExist == False:
        os.mkdir(path)
    path = f'./datasets/{name_of_fol}/images'
    isExist = os.path.exists(path)  
    if isExist == False:
        os.mkdir(path)
    path = f'./datasets/{name_of_fol}/images/val'
    isExist = os.path.exists(path)  
    if isExist == False:
        os.mkdir(path)
    save_image(img, f'./datasets/{name_of_fol}/images/val/{name}')
    path = f'./datasets/{name_of_fol}/labels'
    isExist = os.path.exists(path)  
    if isExist == False:
        os.mkdir(path)
    path = f'./datasets/{name_of_fol}/labels/val'
    isExist = os.path.exists(path)  
    if isExist == False:
        os.mkdir(path)
    shutil.copy(f'./datasets/COCO2017/labels/val2017/{name[:-4]}.txt',path)
    
    
def save_visualize_image(img, name_of_fol ,name, loss_value):
    path = f'./img_results/{name_of_fol}'
    isExist = os.path.exists(path)
    if isExist == False:
        os.mkdir(path)
    path = f'./img_results/{name_of_fol}/images_full'
    isExist = os.path.exists(path)  
    if isExist == False:
        os.mkdir(path)
    path = f'./img_results/{name_of_fol}/images_full/val'
    isExist = os.path.exists(path)  
    if isExist == False:
        os.mkdir(path)
    save_image(img, f'./img_results/{name_of_fol}/images_full/val/{name[:-4]}_{loss_value}{name[-4:]}')
    path = f'./img_results/{name_of_fol}/labels'
    isExist = os.path.exists(path)  
    if isExist == False:
        os.mkdir(path)
    path = f'./img_results/{name_of_fol}/labels/val/'
    isExist = os.path.exists(path)  
    if isExist == False:
        os.mkdir(path)
 
    #shutil.copy(f'./datasets/MOT20Det/train/MOT20-01/output/{name[:-4]}.txt',path + f'{name[:-4]}_{loss_value}.txt')
    
    
def save_visualize_loss_segment(img, name_of_fol ,name, loss_value,number_people,i):
    path = f'./datasets/{name_of_fol}'
    isExist = os.path.exists(path)
    if isExist == False:
        os.mkdir(path)
    path = f'./datasets/{name_of_fol}/images'
    isExist = os.path.exists(path)  
    if isExist == False:
        os.mkdir(path)
    path = f'./datasets/{name_of_fol}/images/val'
    isExist = os.path.exists(path)  
    if isExist == False:
        os.mkdir(path)
    save_image(img, f'./datasets/{name_of_fol}/images/val/{name[:-4]}_{loss_value}_{number_people}_{i}{name[-4:]}')
    path = f'./datasets/{name_of_fol}/labels'
    isExist = os.path.exists(path)  
    if isExist == False:
        os.mkdir(path)
    path = f'./datasets/{name_of_fol}/labels/val/'
    isExist = os.path.exists(path)  
    if isExist == False:
        os.mkdir(path)
    shutil.copy(f'./datasets/COCO2017/labels/val2017/{name[:-4]}.txt',path + f'{name[:-4]}_{loss_value}_{number_people}_{i}.txt')
       
    
    
def full_frame(width=None, height=None):
    mpl.rcParams['savefig.pad_inches'] = 0
    figsize = None if width is None else (width, height)
    fig = plt.figure(figsize=figsize)
    ax = plt.axes([0,0,1,1], frameon=False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.autoscale(tight=True)
    
    
def show_imgs_tensor(img):
    full_frame()
    plt.imshow(img[0].squeeze(0).permute(1, 2, 0).cpu().numpy()/255)
    
def show_imgs_3_channel_tensor(img):
    full_frame()
    plt.imshow(img[0].permute(1, 2, 0).cpu().numpy()/255)
    
    
def check_gray(batch_tensor):
    img1 = batch_tensor[0,0,:,:]
    img2 = batch_tensor[0,1,:,:]
    img3 = batch_tensor[0,2,:,:]
    if torch.all(img1 == img2) and torch.all(img1 == img3):
        return True
    return False


def cut_images(shape, new_shape = [640,640]):
    x,y,w,h = 0.5,0.5,1,1
    gt = Instances(np.array([x, y, w, h]),bbox_format= 'xywh')
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    #r = min(r, 1.0)
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    dw /= 2  # divide padding into 2 sides
    dh /= 2
    ratio = r, r  # width, height ratios
    gt.convert_bbox(format='xyxy')
    gt.denormalize(*shape[:2][::-1])
    gt.scale(*ratio)
    gt.add_padding(dw, dh)
    bbox_after = gt._bboxes.bboxes.reshape(-1).astype(int)
    return bbox_after

def scale_masks(masks, im0_shape, ratio_pad=None):
    """
    Takes a mask, and resizes it to the original image size

    Args:
      masks (torch.Tensor): resized and padded masks/images, [h, w, num]/[h, w, 3].
      im0_shape (tuple): the original image shape
      ratio_pad (tuple): the ratio of the padding to the original image.

    Returns:
      masks (torch.Tensor): The masks that are being returned.
    """
    # Rescale coordinates (xyxy) from im1_shape to im0_shape
    masks = masks.detach().cpu().numpy()
    im1_shape = masks.shape
    if im1_shape[:2] == im0_shape[:2]:
        return masks
    if ratio_pad is None:  # calculate from im0_shape
        gain = min(im1_shape[0] / im0_shape[0], im1_shape[1] / im0_shape[1])  # gain  = old / new
        pad = (im1_shape[1] - im0_shape[1] * gain) / 2, (im1_shape[0] - im0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]
    top, left = int(pad[1]), int(pad[0])  # y, x
    bottom, right = int(im1_shape[0] - pad[1]), int(im1_shape[1] - pad[0])

    if len(masks.shape) < 2:
        raise ValueError(f'"len of masks shape" should be 2 or 3, but got {len(masks.shape)}')
    masks = masks[top:bottom, left:right]
    # masks = masks.permute(2, 0, 1).contiguous()
    # masks = F.interpolate(masks[None], im0_shape[:2], mode='bilinear', align_corners=False)[0]
    # masks = masks.permute(1, 2, 0).contiguous()
    masks = cv2.resize(masks, (im0_shape[1], im0_shape[0]))
    if len(masks.shape) == 2:
        masks = masks[:, :, None]
    masks  = torch.tensor(masks, device="cuda:0").float()
    masks = masks.permute(2, 0, 1).contiguous()
    return masks

def prepare_cloud_folder():
    
    
    path = f'./camera_attack'
    isExist = os.path.exists(path)
    if isExist:
       shutil.rmtree('./camera_attack')
       
    path = f'./img_results/cloud_attack_realsens'
    isExist = os.path.exists(path)
    if isExist:
       shutil.rmtree('./img_results/cloud_attack_realsens')
       
    path = f'./camera_attack/images/val'
    os.makedirs(path)
    path = f'./camera_attack/labels/val'
    os.makedirs(path)

    
    return 


def grad_attack_iter(grad_attack, min_value, max_value,step):
    alpha = torch.zeros_like(grad_attack)
    alpha += grad_attack
    while torch.max(grad_attack) < max_value and torch.min(grad_attack) > min_value:
            grad_attack += alpha*step
    return grad_attack


def grad_attack_each_iter(alpha,step):
    grad_attack = torch.zeros_like(alpha)
    grad_attack += step*alpha
    return grad_attack


def grad_attack_problem_solving(img_folder, image_name):
    trainer = DetectionTrainer('ultralytics/yolo/cfg/my_config.yaml')

    trainer.setup_model()
    trainer.set_model_attributes()
    trainer.model.train()
    model = YOLO('yolov8x.pt')
    loader = create_dataloader(os.path.join(img_folder,image_name),640, 1,32)
    
    for i, batch in enumerate(loader[0]):
        name_img = loader[1].im_files[i]
        x,y  =  loader[1].shapes[i]
        batch_tensor = torch.Tensor(batch['img'].float())

        results = model(batch_tensor)
        mask = torch.zeros_like(batch_tensor)
        
        for bbox in results[0].boxes.xyxy:
            x1,y1, x2, y2 = bbox
            x1,y1, x2, y2 = int(round(float(x1))), int(round(float(y1))),int(round(float(x2))),int(round(float(y2)))
            mask[:,:,y1:y2, x1:x2] = 1

    return batch_tensor, mask



def setup_folder(name_folder):
    path = f'./datasets/{name_folder}'
    isExist = os.path.exists(path)
    if isExist == False:
        os.mkdir(path)
    
    path = f'./datasets/{name_folder}'
    isExist = os.path.exists(path)
    if isExist == False:
        os.mkdir(path)
        
    path = f'./datasets/{name_folder}/images'
    isExist = os.path.exists(path)
    if isExist == False:
        os.mkdir(path)
        
    path = f'./datasets/{name_folder}/images/val'
    isExist = os.path.exists(path)
    if isExist == False:
        os.mkdir(path)
    
    path = f'./datasets/{name_folder}/images/train'
    isExist = os.path.exists(path)
    if isExist == False:
        os.mkdir(path)
        
        
    shutil.copy('./datasets/coco8/images/train/000000000009.jpg',path)
    path = f'./datasets/{name_folder}/labels'
    isExist = os.path.exists(path)
    if isExist == False:
        os.mkdir(path)
        
    path = f'./datasets/{name_folder}/labels/val'
    isExist = os.path.exists(path)
    if isExist == False:
        os.mkdir(path)
        
    path = f'./datasets/{name_folder}/labels/train'
    isExist = os.path.exists(path)
    if isExist == False:
        os.mkdir(path)
        
    shutil.copy('./datasets/coco8/labels/train/000000000009.txt',path)
    
def copy_images_to_folder(name_image, name_folder_src, name_folder_dst):
    src = f'./datasets/{name_folder_src}/images/val'
    dst = f'./datasets/{name_folder_dst}/images/val/'
    shutil.copy(os.path.join(src,name_image),dst)
    name_labels = name_image.split('_')[0]
    path_labels = f'./datasets/COCO2017/labels/val2017/{name_labels}.txt'
    labels_dst = f'./datasets/{name_folder_dst}/labels/val/{name_image[:-4]}.txt'
    shutil.copy(path_labels,labels_dst)
    
    
def plot_loss_log(loss_log, name_fake_img):
    x = np.arange(0,len(loss_log), 1)
    y = np.array(loss_log)
    plt.plot(x,y)
    plt.title(f'{name_fake_img}')
    plt.savefig(f'./attack_result/recent_loss_visual_{name_fake_img}.jpg')
    
def insert(originalfile,string):
    with open(originalfile,'r') as f:
        with open('newfile.txt','w') as f2: 
            f2.write(string)
            f2.write(f.read())
    os.remove(originalfile)
    os.rename('newfile.txt',originalfile)
    
def set_state(state):
    with open('./ultralytics/datasets/haha.yaml') as f:
        doc = yaml.safe_load(f)

    doc['path'] = f'../datasets/{state}'

    with open(f'./ultralytics/datasets/{state}.yaml', 'w') as f:
        yaml.dump(doc, f)  
        

def calculate_loss_of_tensor(tensor, labels):
    trainer = SegmentationTrainer('ultralytics/yolo/cfg/my_config.yaml')

    trainer.setup_model()
    trainer.set_model_attributes()
    trainer.model.train()
    tensor = torch.Tensor(tensor.float()).cpu()
    model = YOLO('yolov8x-seg.pt')
    b = trainer.model(tensor)
    
    for i in range(3):
        b[i].requires_grad_()
        b[i].retain_grad()
    loss_fn = Loss(trainer.model)
    loss, loss_each_module = loss_fn(b, labels)
    loss.retain_grad()
    loss.backward()
    return tensor.grad

########################## MOT20 ###############################################
def coco_to_yolo(img_path, x1, y1, w, h):
    img = cv2.imread(img_path)
    image_h = img.shape[0]
    image_w = img.shape[1]
    x_center = (x1 + w/2) / image_w
    y_center = (y1 + h/2) / image_h
    return [x_center , y_center, w/image_w, h/image_h]

def preprocess_label_MOT20():
    num_frames = 430
    lst_cols = ['class','x','y','width','height']
    gt_path = './gt/gt.txt'
    #Read data
    df = pd.read_csv(gt_path, sep=',', header=None, names=['frame_id','person_id','x','y','width','height','col1','class_id','conf'])
    for i in range(1, num_frames):
    
        temp_str = '000000'
    
        f_str = temp_str[:-len(str(i))]+ str(i) + '.txt'
        img_path = './img1/'+ f_str[:-4]+ '.jpg'
        test_df = df[(df['frame_id']==i) & (df['class_id']==1)]
        test_df['x'] = test_df[['x']].apply(lambda x: coco_to_yolo(img_path, test_df['x'],test_df['y'],test_df['width'], test_df['height'])[0])
        test_df['y'] = test_df[['y']].apply(lambda x: coco_to_yolo(img_path, test_df['x'],test_df['y'],test_df['width'], test_df['height'])[1])
        test_df['width'] = test_df[['width']].apply(lambda x: coco_to_yolo(img_path, test_df['x'],test_df['y'],test_df['width'], test_df['height'])[2])
        test_df['height'] = test_df[['height']].apply(lambda x: coco_to_yolo(img_path, test_df['x'],test_df['y'],test_df['width'], test_df['height'])[3])
        final_df = test_df[['x','y','width','height']]
        final_df['class'] = 0
        final_df = final_df[lst_cols]
        #Create output folder before running
        path = './output'
        isExist = os.path.exists(path)
        if isExist == False:
            os.mkdir(path)
        final_df.to_csv(f'./output/{f_str}',sep=' ', header=False, index = False)   
        
        
def compute_normalized_cross_correlation(image1, image2):
    image1 = image1.squeeze(0)
    image2 = image2.squeeze(0)
    image1 = torch.permute(image1, (1, 2, 0))
    image2 = torch.permute(image2, (1, 2, 0))
    image1 = image1.detach().cpu().numpy()
    image2 = image2.detach().cpu().numpy()
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    gray1 = gray1.astype(np.float32)
    gray2 = gray2.astype(np.float32)
    mean1 = np.mean(gray1)
    mean2 = np.mean(gray2)
    gray1 -= mean1
    gray2 -= mean2
    std1 = np.std(gray1)
    std2 = np.std(gray2)
    ncc = (np.sum(gray1 * gray2) / (std1 * std2 * gray1.size)) - 1e-6
    return ncc 

def delete_temp_img(folder_result):
    path = f'./img_results/{folder_result}/images_full/val/*_*_*.jpg'
    path_mean = f'./img_results/{folder_result}/images_full/val/*_ncc_mean_*.jpg'
    base_path = sorted(glob(path))
    mean_path = sorted(glob(path_mean))
    for path_img in mean_path:
        base_path.remove(path_img)
    for path_img in base_path:
        os.remove(path_img)
        
def delete_temp_video(folder_result):
    path = f'./{folder_result}/images_full/val/*_*_*.jpg'
    path_mean = f'./{folder_result}/images_full/val/*_ncc_mean_*.jpg'
    base_path = sorted(glob(path))
    mean_path = sorted(glob(path_mean))
    for path_img in mean_path:
        base_path.remove(path_img)
    for path_img in base_path:
        os.remove(path_img)
        




def delete_min_object_iter(folder_result, name_img, num_object_min, num_iter_min):
    path_ta = f'./img_results/{folder_result}/images_full/val/*'
    data_lst = sorted(glob(path_data))

    full_lst = []
    for img in data_lst:
        
        
        if(name_img[:-4] == os.path.split(img)[-1].split('_')[0]) and os.path.split(img)[-1].split('_')[1] != 'base.jpg':
            
            if (int(os.path.split(img)[-1].split('_')[1]) == num_object_min and int(os.path.split(img)[-1].split('_')[2][:-4]) == num_iter_min):
                continue
            full_lst.append(img)

        

    for path_img in full_lst:
        os.remove(path_img)


