import random
import numpy as np
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.models.detection import maskrcnn_resnet50_fpn
import Transformation_group as Transformation_group
from PIL import Image
transformation_groups = Transformation_group.transformation_groups
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RoI_model = maskrcnn_resnet50_fpn(pretrained=True)
RoI_model.to(device)
RoI_model.eval() 

def get_transformations(active_groups):
    selected_transformations = []
    for group in active_groups:
        selected_transformations.extend(transformation_groups[group])
    return selected_transformations

def random_choose(transformation_list):
    idx = random.randint(0, len(transformation_list) - 1)
    return transformation_list[idx]

def local_global_augmentation(image_batch, active_group):
    actif_group = get_transformations(active_group)
    first_augmentations = random_choose(actif_group)
    second_augmentations = first_augmentations
    first_batch = []
    second_batch = []
    image_batch = image_batch.to(device)
    with torch.no_grad():
        predicted_roi = RoI_model(image_batch)
    image_batch_convert = image_batch.cpu().numpy()
    for index, image in enumerate(image_batch_convert):
        boxes = predicted_roi[index]['boxes'].cpu().numpy()
        masks = predicted_roi[index]['masks'].cpu().numpy()
        scores = predicted_roi[index]['scores'].cpu().numpy()
        full_image = image_batch[index].permute(1, 2, 0)
        full_transformed_image = first_augmentations(Image.fromarray((full_image.cpu().numpy() * 255).astype(np.uint8)))
        full_transformed_image = np.array(full_transformed_image) / 255.0
        image_permuted = image.transpose(1, 2, 0)
        image_permuted_copy = image_permuted.copy()
        for i,box in enumerate(boxes):
            if scores[i] > 0.5:
              x_1, y_1, x_2, y_2 = box.astype(int)
              mask = masks[i,0] > 0.5
              region = image_permuted[y_1:y_2, x_1:x_2].copy()
              roi = Image.fromarray((region*255).astype(np.uint8),mode="RGB")
              transformed_roi = second_augmentations(roi)
              transformed_roi = np.array(transformed_roi) / 255
              mask_resized = mask[y_1:y_2, x_1:x_2]
              image_permuted_copy[y_1:y_2, x_1:x_2][mask_resized] = transformed_roi[mask_resized]
        
        image_with_transform_roi =  image_permuted_copy.transpose(2, 0, 1)
        second_batch.append(image_with_transform_roi)
        full_transformed_image = torch.tensor(full_transformed_image).permute(2, 0, 1).float()
        first_batch.append(full_transformed_image)
    second_batch= torch.tensor(np.array(second_batch)).float().to(device)
    first_batch = torch.stack(first_batch).float().to(device)
  
    return first_batch,second_batch


def global_global_augmentation(image_batch, active_group):
    first_batch = []
    second_batch = []
    actif_group = get_transformations(active_group)
    first_augmentations = random_choose(actif_group)
    second_augmentations = random_choose(actif_group)
    for image in image_batch:
        first_image = first_augmentations(image)
        second_image = second_augmentations(image)
        first_batch.append(first_image)
        second_batch.append(second_image)
    
    return torch.stack(first_batch), torch.stack(second_batch)


def local_local_augmentation(batch_image, active_groups):
    
    selected_transformations = get_transformations(active_groups)
    transform_batch_1 = random_choose(selected_transformations)
    transform_batch_2 = random_choose(selected_transformations)
    image_batch = image_batch.to(device)
    with torch.no_grad():
        predicted_roi = RoI_model(image_batch)

    batch_image = np.array(batch_image.cpu())
    for index, image in enumerate(batch_image):
        boxes = predicted_roi[index]['boxes'].cpu().numpy()
        masks = predicted_roi[index]['masks'].cpu().numpy()
        scores =predicted_roi[index]['scores'].cpu().numpy()
        
        for i, box in enumerate(boxes):
            if scores[i] > 0.5:
                x1, y1, x2, y2 = box.astype(int)
                mask = masks[i, 0] > 0.50  # Obtention du masque
                image_permuted = image.transpose(1, 2, 0) 
                colored_object = image_permuted[y1:y2, x1:x2].copy()
                colored_object = Image.fromarray((colored_object * 255).astype(np.uint8), mode='RGB')
                colored_object_transformed_1 = transform_batch_1(colored_object)
                colored_object_transformed_2 = transform_batch_1(colored_object)
                colored_object_transformed_1 = np.array(colored_object_transformed_1) / 255.0
                colored_object_transformed_2 = np.array(colored_object_transformed_2) / 255.0
                colored_object_transformed_1 = Image.fromarray((colored_object_transformed_1 * 255).astype(np.uint8))
                colored_object_transformed_2 = Image.fromarray((colored_object_transformed_2 * 255).astype(np.uint8))
                colored_object_transformed_1 = colored_object_transformed_1.resize((x2 - x1, y2 - y1))
                colored_object_transformed_2 = colored_object_transformed_2.resize((x2 - x1, y2 - y1))
                colored_object_transformed_1 = np.array(colored_object_transformed_1) / 255.0
                colored_object_transformed_2 = np.array(colored_object_transformed_2) / 255.0
                image_permuted_1 = image_permuted.copy()
                image_permuted_2 = image_permuted.copy()
                mask_resized = mask[y1:y2, x1:x2]  
                mask_resized = np.expand_dims(mask_resized, axis=-1)  
                mask_resized = np.repeat(mask_resized, 3, axis=-1)  
                image_permuted_1[y1:y2, x1:x2][mask_resized] = colored_object_transformed_1[mask_resized]
                image_permuted_2[y1:y2, x1:x2][mask_resized] = colored_object_transformed_2[mask_resized]
                image_clipped_1 = np.clip(image_permuted_1, 0, 1)
                image_clipped_2 = np.clip(image_permuted_2, 0, 1)
                image_to_save_1 = image_permuted_1.transpose(2, 0, 1) 
                image_to_save_2 = image_permuted_2.transpose(2, 0, 1)                
                first_batch.append(image_to_save_1)
                second_batch.append(image_to_save_2)  
    first_batch = np.array(first_batch)
    second_batch = np.array(second_batch)
    first_batch = torch.from_numpy(first_batch)
    second_batch = torch.from_numpy(second_batch)
    return first_batch, second_batch


        




