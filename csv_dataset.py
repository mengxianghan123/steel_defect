import numpy as np
import os
import cv2
from tqdm import tqdm

PATH = 'train.csv'
PATH_MASK = 'mask'

content = np.loadtxt(PATH,str,delimiter=',',skiprows=1)
# print(content)

'''
for row in tqdm(content):
    img_id = row[0]
    mask_class = int(row[1])
    mask_describe = row[2]
    mask_describe = mask_describe.split(' ')    #list
    # print(mask_describe)

    mask_path = os.path.join(PATH_MASK, row[0])
    mask = np.zeros((256, 1600), dtype=np.uint8)

    for idx, element in enumerate(mask_describe):
        if idx % 2 == 0:
            pixel_start = int(mask_describe[idx])
            pixel_len = int(mask_describe[idx + 1])
            start_col = pixel_start//256
            start_row = pixel_start - start_col * 256
            mask[start_row : start_row + pixel_len, start_col] = mask_class

    # print()
    cv2.imwrite(mask_path, mask)
    # break
    '''
img_all = os.listdir('train_images')
for img in tqdm(img_all):
    if img not in content[:,0]:
        mask = np.zeros((256, 1600), dtype=np.uint8)
        cv2.imwrite(os.path.join(PATH_MASK,img), mask)
