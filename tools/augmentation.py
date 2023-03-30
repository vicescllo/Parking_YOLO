
import os
import cv2
import numpy as np
import albumentations as alb
import json

class Augmentation:
    """ 
    This class is used to augment images, in this case we are using albumentations library
    for simulate weather conditions in the parking dataset. 

    Args:
        image_filename (str): The name of the image file.
        label_filename (str): The name of the label file.
    """


    def __init__(self, augmented_images_path, augmented_labels_path):
        self.augmented_images_path = augmented_images_path
        self.augmented_labels_path = augmented_labels_path

    
    def load(self):
        img = cv2.imread(self.augmented_images_path)
        height_image, width_image, _  = img.shape
        augmentor = alb.Compose([
            alb.RandomSnow(brightness_coeff=2.5, snow_point_lower=0.3, snow_point_upper=0.5, p=0.1), #To simulate snow
            alb.RandomRain(brightness_coefficient=0.9, drop_width=1, blur_value=5, p=0.2), #To simulate rain
            alb.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), angle_lower=0.5, p=0.3), #To simulate sun flare
            alb.RandomShadow(num_shadows_lower=1, num_shadows_upper=1, shadow_dimension=5, shadow_roi=(0, 0.5, 1, 1), p=0.25), #To simulate shadows
            alb.RandomFog(fog_coef_lower=0.7, fog_coef_upper=0.8, alpha_coef=0.1, p=0.1), #To simulate fog
            alb.CenterCrop(height=500, width=500, p=0.1), #To crop the image, in this case we are using 500x500
            

        ], bbox_params=alb.BboxParams(format='yolo', label_fields=['class_labels']))

        with open(self.augmented_labels_path, "r") as label_file:
            label = label_file.read()
        width_image=img.shape[1]
        height_image=img.shape[0]
        label = label.split('\n')[:-1]
        label = [l.split(' ') for l in label]
        label = [[int(l[0]), float(l[1]), float(l[2]), float(l[3]), float(l[4])] for l in label]
       
        coords=label
        x=[i[1:] for i in coords]
        y=[i[0] for i in coords]
        try:
            augmented = augmentor(image=img, bboxes=x, class_labels=y)
            #Convert tupla to list
            augmented_image = augmented['image']
            augmented_label = augmented['bboxes']
            augmented_labels = augmented['class_labels']
            augmented_label = [list(i) for i in augmented_label]
            augmented_label = [[augmented_labels[i]] + augmented_label[i] for i in range(len(augmented_label))]
            cv2.imwrite(self.augmented_images_path, augmented_image)
        
            with open(self.augmented_labels_path, "w") as augmented_label_file:
                for elem in augmented_label:
                    linea = " ".join(str(round(x, 6)) for x in elem)
                    augmented_label_file.write(linea + "\n")
        except:
            print('error')
            pass
        
