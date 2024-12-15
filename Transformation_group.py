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
        transforms.ColorJitter(brightness=0.6, hue=0.2),
        transforms.ColorJitter(contrast=0.6, saturation=0.6),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, hue=0.1),
        transforms.ColorJitter(saturation=0.5, brightness=0.5),
        transforms.ColorJitter(contrast=0.2, hue=0.4),
        transforms.ColorJitter(brightness=0.5, saturation=0.2, hue=0.2),
        transforms.ColorJitter(contrast=0.3, saturation=0.4),
        transforms.ColorJitter(brightness=0.1, hue=0.1),
        transforms.ColorJitter(saturation=0.7, hue=0.2),
        transforms.ColorJitter(brightness=0.2, contrast=0.1, hue=0.3),
        transforms.ColorJitter(saturation=0.6, hue=0.4),
        transforms.ColorJitter(brightness=0.3, contrast=0.5, hue=0.2),
    ],

    "GaussianBlur": [
        transforms.GaussianBlur(kernel_size=(3, 1), sigma=(0.1, 5)),  # Flou gaussien
        transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2)),
        transforms.GaussianBlur(kernel_size=(1, 1), sigma=(0.1, 3)),
        transforms.GaussianBlur(kernel_size=(1, 3), sigma=(0.1, 4)),
        transforms.GaussianBlur(kernel_size=(3, 1), sigma=(0.5, 1)),
        transforms.GaussianBlur(kernel_size=(3, 1), sigma=(0.2, 0.8)),
        transforms.GaussianBlur(kernel_size=(3, 1), sigma=(0.2, 1)),
        transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.3, 2)),
        transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 1)),
        transforms.GaussianBlur(kernel_size=(5, 3), sigma=(0.3, 3)),
        transforms.GaussianBlur(kernel_size=(7, 7), sigma=(0.1, 2.5)),
        transforms.GaussianBlur(kernel_size=(3, 5), sigma=(0.2, 2)),
        transforms.GaussianBlur(kernel_size=(1, 5), sigma=(0.5, 1.5)),
        transforms.GaussianBlur(kernel_size=(5, 1), sigma=(0.4, 3)),
        transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 0.5)),
        transforms.GaussianBlur(kernel_size=(3, 5), sigma=(0.3, 1.8)),
        transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.2, 1.2)),
        transforms.GaussianBlur(kernel_size=(1, 3), sigma=(0.3, 1.4)),
        transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.4, 2)),
        transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.5, 3)),
    ]
}
