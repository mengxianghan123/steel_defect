import numpy as np
import os
import shutil
from tqdm import tqdm
import glob
import cv2


def get_val_dataset():
    img_all = os.listdir('train_images')
    num_img_all = len(img_all)

    num_img_val = 1000
    # img_idx = np.random.randint(low=num_img_all, size=num_img_val)
    img_idx = np.random.choice(num_img_all, num_img_val, replace=False)

    content = np.loadtxt('train.csv', str, delimiter=',', skiprows=1)

    for i in tqdm(img_idx):
        old_path = os.path.join('train_images', img_all[i])
        new_path = os.path.join('val_dataset', img_all[i])
        shutil.copy(old_path, new_path)

        content_row = np.where(content[:, 0] == img_all[i])[0]
        if len(content_row) == 0:
            pass
        else:
            for row in content_row:
                save_txt(
                    img=img_all[i], label=content[row, 1], rle=content[row, 2])

    return img_idx


def save_txt(img, label, rle):
    with open('train_dataset/'+img[:-4]+'.txt', 'a') as f:
        # print(img)
        # print(label)
        # print(rle)

        f.writelines(label+'\n')
        f.writelines(rle+'\n')


def get_train_dataset():
    val_img_list = glob.glob('val_dataset/*.jpg')
    for i in range(len(val_img_list)):
        val_img_list[i] = val_img_list[i][12:]

    content = np.loadtxt('train.csv', str, delimiter=',', skiprows=1)

    for idx, img in enumerate(content[:, 0]):
        if img not in val_img_list:
            if img not in os.listdir('train_dataset'):
                shutil.copy(os.path.join('train_images', img),
                            os.path.join('train_dataset', img))
            save_txt(img, content[idx, 1], content[idx, 2])


def show_the_mask(path, num):
    img_list = glob.glob(os.path.join(path, '*.jpg'))
    num_cur = 0
    for img_full_name in img_list:
        if num_cur == num:
            return 0
        img_name = img_full_name[:-4]
        txt_full_name = img_name + '.txt'
        masks = []

        if txt_full_name.split('/')[-1] in os.listdir(txt_full_name.split('/')[0]):

            labels, masks = txtdecode(txt_full_name)  

        img = cv2.imread(img_full_name)
        for mask in masks:
            img[mask == 1] = [0, 255, 0]
        cv2.imwrite('visualize/' + img_name.split('/')[-1] + '.jpg', img)
        num_cur = num_cur + 1
        


def txtdecode(txt_full_name):
    with open(txt_full_name, 'r') as f:
        content = np.array(f.readlines())
        row = list(range(len(content)))
        labels = content[row[::2]]
        for i in range(len(labels)):
            labels[i] = labels[i][:-1]  # labels是个列表

        rles = content[row[1::2]]
        masks = []
        for i in range(len(rles)):
            mask = rle2mask(rles[i][:-1])
            masks.append(mask)
    return labels, masks


def rle2mask(rle):
    rle = rle.split(' ')
    mask = np.zeros(256*1600, dtype=np.uint8)
    for x, element in enumerate(rle):
        if x % 2 == 0:
            pixel_start = int(rle[x])
            pixel_len = int(rle[x + 1])
            mask[pixel_start - 1:pixel_start+pixel_len-1] = 1
    mask = np.reshape(mask, (256, 1600), 'F')
    return mask


if __name__ == "__main__":
    # get_val_dataset()
    get_train_dataset()
    # show_the_mask('val_dataset', 10)
