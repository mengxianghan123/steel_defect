import numpy as np
import cv2
import torch
from torchvision import transforms
import os

file_list = os.listdir('train_images')
for file in file_list:
    img = cv2.imread(os.path.join('train_images', file))
    if (img.shape!=(256,1600,3)):
        print(file)