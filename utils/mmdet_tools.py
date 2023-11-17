import torch
import torchvision
import matplotlib.pyplot as plt

def inverse_to_base(image):
    inv_normalize = torchvision.transforms.Normalize(
            mean=[-123.675/58.395, -116.28/57.12, -103.53/57.375],
            std=[1/(58.395/255), 1/(57.12/255), 1/(57.375/255)]
        )
    inv_tensor = inv_normalize(image)
    return inv_tensor

def compute_normalized_cross_correlation(image1, image2):
    image1 = image1.squeeze(0)
    image2 = image2.squeeze(0)
    
    gray1  = torchvision.transforms.Grayscale()(image1).squeeze(0).float()
    gray2  = torchvision.transforms.Grayscale()(image2).squeeze(0).float()
    
    mean1 = torch.mean(gray1)
    mean2 = torch.mean(gray2)
    
    gray1 -= mean1
    gray2 -= mean2
    std1 = torch.std(gray1)
    std2 = torch.std(gray2)
    
    ncc = (torch.sum(gray1 * gray2) / (std1 * std2 * gray1.shape[0]*gray1.shape[1])) - 1e-6
    return ncc 

def save_loss(loss_hist):
    plt.savefig()