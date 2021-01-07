import numpy as np
import cv2
import torch
from torchvision import transforms
import os
import glob
from sklearn.metrics import confusion_matrix

# file_list = glob.glob('train_dataset/*.txt')
# for file in file_list:
#     with open(file, 'r') as f:
#         con = np.array(f.readlines())
#         row = list(range(0, len(con), 2))
#         if '1\n' in con[row] or '2\n' in con[row] or '4\n' in con[row]:
#             print(file)
a = np.array([0])
b = np.array([0,0,0,0,0,0,0,0])
# cm = confusion_matrix(a,b,labels=[1])

print(a is np.array([0]))

