# Sample code from the TorchVision 0.3 Object Detection Finetuning Tutorial
# http://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

import os
import numpy as np
import torch
from PIL import Image
import cv2

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from engine import train_one_epoch, evaluate
import utils
import transforms as T
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import albumentations as A


class SteelDataset(object):
    def __init__(self, root, train):
        self.root = root
        self.train = train
        # load all image files, sorting them to
        # ensure that they are aligned
        self.content = np.loadtxt('train.csv', str, delimiter=',', skiprows=1)
        self.imgs = list(sorted(np.unique(self.content[:, 0])))
        # self.imgs = list(sorted(os.listdir(os.path.join(root, "train_images"))))

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "train_images", self.imgs[idx])
        img = Image.open(img_path).convert("RGB")

        id = np.where(self.content[:, 0] == self.imgs[idx])[0]
        # print(id)
        boxes = []
        masks = []
        labels = []
        area = []
        image_id = torch.tensor([idx])
        iscrowd = []
        if len(id) == 0:
            pass
        else:
            for row in id:
                mask_describe = self.content[row, 2]
                mask_describe = mask_describe.split(' ')  # list
                mask = np.zeros(256 * 1600, dtype=np.uint8)
                mask_cpy = mask.copy()
                for x, element in enumerate(mask_describe):
                    if x % 2 == 0:
                        pixel_start = int(mask_describe[x])
                        pixel_len = int(mask_describe[x + 1])
                        mask_cpy[pixel_start - 1:pixel_start+pixel_len-1] = 255
                mask_cpy = np.reshape(mask_cpy, (256, 1600), 'F')

                cont, _ = cv2.findContours(
                    mask_cpy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # print(len(cont))
                for i in range(len(cont)):
                    mask_res = np.zeros((256, 1600))
                    cv2.drawContours(mask_res, cont, i, 1, -1)

                    pos = np.where(mask_res)
                    xmin = np.min(pos[1])
                    xmax = np.max(pos[1])
                    ymin = np.min(pos[0])
                    ymax = np.max(pos[0])
                    if (xmin == xmax or ymin == ymax):

                        continue
                    masks.append(mask_res)
                    labels.append(int(self.content[row, 1]))
                    boxes.append([xmin, ymin, xmax, ymax])

            img, boxes, labels, masks = get_transform(train=self.train,
                                                      img=img,
                                                      boxes=boxes,
                                                      labels=labels,
                                                      masks=masks)

            for box in boxes:
                area.append((box[3]-box[1])*(box[2]-box[0]))
                iscrowd.append(0)

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

        # print(target)
        return img, target

    def __len__(self):
        return len(self.imgs)


def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(
        pretrained=False)
    # model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model


def get_transform(train, img, boxes, labels, masks):
    if train:
        trans = A.Compose([A.Blur(),
                           A.VerticalFlip(),
                           A.HorizontalFlip(p=0.5),
                           A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, always_apply=False, p=0.5),
                           A.GaussNoise(var_limit=(10.0, 50.0), mean=0, always_apply=False, p=0.5),
                           A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), always_apply=False, p=0.5),
                           A.Cutout(num_holes=8, max_h_size=8, max_w_size=8, fill_value=0, always_apply=False, p=0.5)],
                           bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_label']))
        img = np.array(img)

        # print(img.shape)
        # print(masks)
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
    else:
        img = np.array(img)
        img = np.transpose(img, (2, 0, 1))
        img = torch.as_tensor(img).div(255)
        return img, boxes, labels, masks


def visualize(img, mask_target, mask):
    img = img.mul(255).cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img_cpy = img.copy()
    # np.array([0,255,0],dtype = np.uint8)
    img_cpy[mask_target != 0] = [0, 255, 0]
    cv2.imwrite('imgtarget.jpg', img_cpy)
    img[mask != 0] = [255, 0, 0]
    cv2.imwrite('img_predict.jpg', img)


def evaluate_dice(model, data_loader):
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)

    cpu_device = torch.device("cpu")
    model.eval()
    dice = 0

    for images, targets in tqdm(data_loader):
        images = list(img.to('cuda') for img in images)
        outputs = model(images)
        # print(outputs)

        masks = outputs[0]['masks'].detach().cpu().numpy()
        labels = outputs[0]['labels'].detach().cpu().numpy()
        scores = outputs[0]['scores'].detach().cpu().numpy()

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
        # break

        cm = confusion_matrix(
            masks_flatten, masks_target_flatten, labels=[1, 2, 3, 4])
        tp = 0
        for i in range(4):
            tp += cm[i, i]
        dice += (tp*2/(409600*2-np.bincount(masks_target_flatten)
                       [0]-np.bincount(masks_flatten)[0]))

    return dice/1000


def main():
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    # print(device)
    num_classes = 5
    # use our dataset and defined transformations
    dataset = SteelDataset('', train=True)
    dataset_test = SteelDataset('', train=False)

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-1000])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-1000:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=8, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes)
    model.load_state_dict(torch.load(
        os.path.join('result1229', 'epoch20.ckpt')))

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=15,
                                                   gamma=0.3)

    # let's train it for 10 epochs
    num_epochs = 200

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader,
                        device, epoch, print_freq=100)
        # update the learning rate
        lr_scheduler.step()

        dice = evaluate_dice(model, data_loader_test)
        # evaluate on the test dataset
        # if (epoch%5==0) and (epoch!=0):
        # if epoch == 0:
        # _, coco = evaluate(model, data_loader_test, device=device)
        # else:
        #     evaluate_dice(model, data_loader_test)
        # evaluate(model, data_loader_test, device=device, coco=coco)
        if (epoch % 2 == 0):
            torch.save(model.state_dict(), 'result15/epoch'+str(epoch)+'.ckpt')
        with open('result15/log.txt', 'a') as f:
            f.writelines('epoch = '+str(epoch)+' : '+str(dice)+'\n')
        # break
    print("That's it!")


if __name__ == "__main__":
    main()
    # dataset = SteelDataset('', get_transform(train=True))
    # img,target = dataset[0]
    # print(target)
