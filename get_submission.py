import torch
import pycocotools
import numpy as np
import cv2
import os
from training import get_model_instance_segmentation
from tqdm import tqdm

def get_model_result(img, path):
    device = 'cuda'
    model = get_model_instance_segmentation(5)
    model.load_state_dict(torch.load(os.path.join('result1229',path)))
    model.to(device)
    model.eval()

    img = cv2.imread(os.path.join('test_images', img))
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = np.transpose(img,(2,0,1))
    img = torch.from_numpy(img).float().div(255)
    img = img.to(device)
    # print(img.shape)
    # print(img)

    target = model([img])
    # print(target)
    masks = target[0]['masks'].detach().cpu().numpy()
    labels = target[0]['labels'].detach().cpu().numpy()

    labels_unq = np.unique(labels)
    masks = np.squeeze(masks,1)
    masks = np.where(masks,1,0)
    masks_final = np.zeros((len(labels_unq),256,1600))
    for i,lab in enumerate(labels_unq):
        idx = np.where(labels==lab)[0]
        # print(idx)

        masks_final[i] = np.sum(masks[idx],axis=0)
        # print(np.unique(masks_final[i]))
        masks_final = np.where(masks_final==0,0,1).astype(np.uint8)

        # print(masks_final)
    return masks_final,labels_unq

def mask2rle(img, width, height):
    rle = []
    lastColor = 0
    currentPixel = 0
    runStart = -1
    runLength = 0

    for x in range(width):
        for y in range(height):
            currentColor = img[y][x]
            if currentColor != lastColor:
                if currentColor == 1:
                    runStart = currentPixel
                    runLength = 1
                else:
                    rle.append(str(runStart))
                    rle.append(str(runLength))
                    runStart = -1
                    runLength = 0
                    # currentPixel = 0
            elif runStart > -1:
                runLength += 1
            lastColor = currentColor
            currentPixel+=1

    return " ".join(rle)
if __name__ == '__main__':
    content = np.loadtxt('sample_submission.csv', str, delimiter=',', skiprows=1)
    final_submission = np.zeros((0,3))
    # print(content)
    for row in tqdm(content):
        masks,labels = get_model_result(row[0],'epoch20.ckpt')

        for i,lab in enumerate(labels):
            img = row[0]
            rle = mask2rle(masks[i],1600,256)
            final_submission = np.append(final_submission, np.array([img,rle,lab])[None, :], axis=0)

    np.savetxt('submission.csv', final_submission, fmt='%s', delimiter=',')
