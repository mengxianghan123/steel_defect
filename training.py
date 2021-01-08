import os
import numpy as np
import torch
from PIL import Image
import cv2
import glob

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from engine import train_one_epoch, evaluate
import utils
import transforms as T
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import albumentations as A
from data_construct import rle2mask, txtdecode


class SteelDataset(object):
    def __init__(self, root): # root不仅决定dataset是train还是val，还决定是否动态数据增强
        self.root = root
        self.imgs = glob.glob(os.path.join(root, '*.jpg')) #list of full_path of img

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        img = Image.open(img_path).convert("RGB")
        txt_path = img_path[:-4]+'.txt'

        # no defect in the img
        if txt_path.split('/')[-1] not in os.listdir(self.root):
            target = {}
            img = np.array(img)
            img = np.transpose(img, (2, 0, 1))
            img = torch.as_tensor(img).div(255)
            return img, target
        
        # defect exists
        boxes = []
        masks = []
        labels = []
        area = []
        image_id = torch.tensor([idx])
        iscrowd = []
        with open(txt_path, 'r') as f:
            content = np.array(f.readlines())
        num_row = len(content)
        label_list = content[list(range(0, num_row, 2))]
        rle_list = content[list(range(1, num_row, 2))]
        for idx, rle in enumerate(rle_list):
            mask = rle2mask(rle[:-1])
            masks, labels, boxes = mask2instance(masks, mask, labels, label_list[idx][:-1], boxes)
        
        for box in boxes:
            area.append((box[3]-box[1])*(box[2]-box[0]))
            iscrowd.append(0)

        # 动态数据增强
        if 1 in labels or 2 in labels or 4 in labels:
            img, boxes, labels, masks = get_transform(img=img,
                                                      boxes=boxes,
                                                      labels=labels,
                                                      masks=masks)
        else:
            img = np.array(img)
            img = np.transpose(img, (2, 0, 1))
            img = torch.as_tensor(img).div(255)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        area = torch.as_tensor(area, dtype=torch.float32)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        # todo normalize
        return img, target

    def __len__(self):
        return len(self.imgs)

def mask2instance(masks, mask, labels, label, boxes):
    cont, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(cont)):
        mask_res = np.zeros((256, 1600))
        cv2.drawContours(mask_res, cont, i, 1, -1)
        pos = np.where(mask_res)
        xmin = np.min(pos[1])
        xmax = np.max(pos[1])
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])
        if (xmax-xmin)*(ymax-ymin)<5:
            continue
        boxes.append([xmin, ymin, xmax, ymax])
        masks.append(mask_res)
        labels.append(int(label))
    return masks, labels, boxes



def get_model_instance_segmentation(num_classes):

    model = torchvision.models.detection.maskrcnn_resnet50_fpn(
        pretrained=False)

    in_features = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256

    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)
    return model


def get_transform(img, boxes, labels, masks):
    trans = A.Compose([ A.Blur(),
                        A.VerticalFlip(),
                        A.HorizontalFlip(p=0.5),
                        # A.Rotate(limit=30, interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, p=0.5),
                        #A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, always_apply=False, p=0.5),
                        #A.GaussNoise(var_limit=(10.0, 50.0), mean=0, always_apply=False, p=0.5),
                        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), always_apply=False, p=0.5)],
                        # A.Cutout(num_holes=8, max_h_size=8, max_w_size=8, fill_value=0, always_apply=False, p=0.5)],
                        bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_label']))
    img = np.array(img)

    masks = np.array(masks)
    transformed = trans(image=img, bboxes=boxes,
                        mask=masks, class_label=labels)
    transformed_img = transformed['image']
    transformed_bboxes = transformed['bboxes']
    transformed_masks = transformed['mask'].copy()
    transformed_labels = transformed['class_label']

    transformed_img = np.transpose(transformed_img, (2, 0, 1))
    transformed_img = torch.as_tensor(transformed_img).div(255)
    return transformed_img, transformed_bboxes, transformed_labels, transformed_masks


def visualize(img, mask_target, mask=None):
    # img = img.mul(255).cpu().numpy()
    # img = np.transpose(img, (1, 2, 0))
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = np.array(img)
    img_cpy = img.copy()
    # np.array([0,255,0],dtype = np.uint8)
    img_cpy[mask_target != 0] = [0, 255, 0]
    cv2.imwrite('visualize/imgtarget.jpg', img_cpy)
    if(mask):
        img[mask != 0] = [255, 0, 0]
        cv2.imwrite('img_predict.jpg', img)


def evaluate_dice(model, data_loader):
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)

    cpu_device = torch.device("cpu")
    model.eval()
    dice_sum = 0

    for images, targets in tqdm(data_loader):

        images = list(img.to('cuda') for img in images)
        outputs = model(images)

        masks = outputs[0]['masks'].detach().cpu().numpy()
        labels = outputs[0]['labels'].detach().cpu().numpy()
        scores = outputs[0]['scores'].detach().cpu().numpy()

        if not targets[0]: # 无瑕疵图
            if len(labels)==0:
                dice_sum += 1
                continue
            else:
                dice_sum += (4 - len(np.unique(labels))) * 0.25
                continue

        if len(labels)==0: # 预测空白图
            lab = targets[0]['labels'].numpy()
            dice_sum += (4 - len(np.unique(lab))) * 0.25
            continue
        
        for i in range(len(labels)):
            mask_label = np.where(masks[i] == 0, 0, labels[i])
            mask_score = np.where(masks[i] == 0, 0, scores[i])
        mask_max = np.argsort(mask_score, axis=0)[-1, :, :]
        row = np.arange(1600)
        col = np.arange(256)
        col, row = np.meshgrid(row, col)
        masks = mask_label[mask_max, row, col]
        masks_flatten = masks.flatten()

        masks_target = targets[0]['masks'].numpy()

        labels_target = targets[0]['labels'].numpy()[:, None, None]

        masks_target = masks_target * labels_target

        masks_target = np.sum(masks_target, axis=0)
        masks_target_flatten = masks_target.flatten()
        # visualize(images[0], masks_target, masks)
        # visualize(images[0], masks_target)
        # break
        dice = 0
        for cls in range(4):
            masks_flatten_cls = np.where(masks_flatten==(cls+1), 1, 0)
            masks_target_flatten_cls = np.where(masks_target_flatten==(cls+1), 1, 0)
            if (len(np.unique(masks_flatten_cls))==1) and (len(np.unique(masks_target_flatten_cls)))==1:
                dice += 1
            elif (len(np.unique(masks_flatten_cls))==1) or (len(np.unique(masks_target_flatten_cls)))==1:
                dice += 0
            else:
                cm = confusion_matrix(
                masks_flatten_cls, masks_target_flatten_cls, labels=[1])
                tp = cm[0,0]
                dice += (tp*2/(409600*2-np.bincount(masks_target_flatten_cls)
                        [0]-np.bincount(masks_flatten_cls)[0]))
        dice = dice/4.0
        dice_sum += dice
    return dice_sum/1000.0


def main():
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    num_classes = 5

    dataset = SteelDataset('train_dataset')
    dataset_test = SteelDataset('val_dataset')

    # split the dataset in train and test set
    # indices = torch.randperm(len(dataset)).tolist()
    # dataset = torch.utils.data.Subset(dataset, indices[:-1000])
    # dataset_test = torch.utils.data.Subset(dataset_test, indices[-1000:])

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=8, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    model = get_model_instance_segmentation(num_classes)
    # model.load_state_dict(torch.load(
    #     os.path.join('result1229', 'epoch20.ckpt')))

    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params=params,lr=0.01,betas=(0.9, 0.999),eps=1e-8)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=20,
                                                   gamma=0.3)

    num_epochs = 200

    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, data_loader,
                        device, epoch, print_freq=100)
        lr_scheduler.step()
        # model.load_state_dict(torch.load(
        # os.path.join('result17', 'epoch'+str(epoch)+'.ckpt')))
        dice = evaluate_dice(model, data_loader_test)
        # if (epoch%5==0) and (epoch!=0):
        # if epoch == 0:
        # _, coco = evaluate(model, data_loader_test, device=device)
        # else:
        #     evaluate_dice(model, data_loader_test)
        # evaluate(model, data_loader_test, device=device, coco=coco)
        if (epoch % 2 == 0):
            torch.save(model.state_dict(), 'result17/epoch'+str(epoch)+'.ckpt')

        with open('result17/log.txt', 'a') as f:
            f.writelines('epoch = '+str(epoch)+' : '+str(dice)+'\n')
        
    print("That's it!")

def dataset_check(num=0):
    dataset = SteelDataset('train_dataset')
    image, target = dataset[num]
    if target:
        image = image.mul(255)
        image = np.array(image)
        image = np.transpose(image,(1,2,0))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        masks_target = target['masks'].numpy()
        labels_target = target['labels'].numpy()[:, None, None]
        masks_target = masks_target * labels_target
        masks_target = np.sum(masks_target, axis=0)
        image[masks_target==1] = [0,255,0]
        image[masks_target==2] = [255,255,0]
        image[masks_target==3] = [0,255,255]
        image[masks_target==4] = [255,0,255]
        cv2.imwrite('visualize/target.jpg', image)
    else:
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite('visualize/target.jpg', image)
    
    img_full_name = dataset.imgs[num]
    img_name = img_full_name[:-4]
    txt_full_name = img_name + '.txt'
    masks = []
    if txt_full_name.split('/')[-1] in os.listdir(txt_full_name.split('/')[0]):
        labels, masks = txtdecode(txt_full_name)  
    img = cv2.imread(img_full_name)
    for ind, mask in enumerate(masks):
        if labels[ind]=='1':
            img[mask == 1] = [0, 255, 0]
        elif labels[ind]=='2':
            img[mask == 1] = [255, 255, 0]
        elif labels[ind]=='3':
            img[mask == 1] = [0, 255, 255]
        elif labels[ind]=='4':
            img[mask == 1] = [255, 0, 255]   
    cv2.imwrite('visualize/ans.jpg', img)
if __name__ == "__main__":
    main()
    # dataset_check(2682)
    


