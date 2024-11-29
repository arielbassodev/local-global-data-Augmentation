
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import random

transformation_groups = {
    "ColorJitter": [
        transforms.ColorJitter(brightness=0.5),  # Variation de luminosité
        transforms.ColorJitter(contrast=0.5),   
        transforms.ColorJitter(saturation=0.5),  
        transforms.ColorJitter(hue=0.3),        
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  
        transforms.ColorJitter(saturation=0.3, hue=0.1),        
        transforms.ColorJitter(brightness=0.4, saturation=0.4), 
        transforms.ColorJitter(contrast=0.4, hue=0.2),
        transforms.GaussianBlur(kernel_size=(3, 1), sigma=(0.1, 5)),  # Flou gaussien                                                         
        transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2)),  
        transforms.GaussianBlur(kernel_size=(1, 1), sigma=(0.1, 3)),  
        transforms.GaussianBlur(kernel_size=(1, 3), sigma=(0.1, 4)),  
        transforms.GaussianBlur(kernel_size=(3, 1), sigma=(0.5, 1)),  
        transforms.GaussianBlur(kernel_size=(3, 1), sigma=(0.2, 0.8)),  
        transforms.GaussianBlur(kernel_size=(3, 1), sigma=(0.2, 1)),  
        transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.3, 2))            
    ],


}
