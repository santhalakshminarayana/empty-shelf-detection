import os
import cv2
import numpy as np
import math
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from MobileNet import Mobile_Net_V2
from Default_Boxes import Default_Boxes
from DSSD_Model import DSSD
from utils import net_loss, non_max_supression, plot_img

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
torch.autograd.set_detect_anomaly(True)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

mob_net_v2 = torch.load('mob_net_v2.pth', device)
_ = mob_net_v2.eval()

default_boxes = Default_Boxes().forward().to(device)
dssd = DSSD(mob_net_v2).to(device)
dssd = torch.load('dssd.pth', device)


def get_batch_ids(n, batch_size = 8):
    ind = np.random.permutation(n).tolist()
    for i in range(0, n, batch_size):
        yield ind[i : i + batch_size]

def read_images(images_path, labels_path):
    X = []
    Y_annot = []
    Y_labels = []
    for img_name in tqdm(os.listdir(images_path)):
        # read images
        name = img_name.split('.')[0]
        ext = img_name.split('.')[1]
        img = cv2.imread(os.path.join(images_path, name + '.' + ext))
        img = img[:, :, ::-1]
        # reshape to (3, dim, dim)
        img_shape = img.shape
        img = img.reshape(3, img_shape[0], img_shape[1])
        img = img / 255.
        X.append(img)

        # read annotations and class labels
        annotation_file = os.path.join(labels_path, name + '.txt')
        with open(annotation_file, 'r') as f:
            annot = []
            obj_class = []
            for line in f.readlines():
                annots = list(map(float, line.split(' ')))
                obj_class.append(int(annots[0]))
                annot.append(annots[1:])
        Y_annot.append(annot)
        Y_labels.append(obj_class)
    X = np.array(X)
    Y_annot = np.array(Y_annot)

    return X, Y_annot

train_img_images = 'train_img/images'
train_img_labels = 'train_img/labels'

test_img_images = 'test_img/images'
test_img_labels = 'test_img/labels'

X, Y_annot = read_images(train_img_images, train_img_labels)
tX, tY_annot = read_images(test_img_images, test_img_labels)

def plot_result(num_samples = 2):
    inds = np.random.permutation(len(tX))[:num_samples]
    for i in inds:
        x = torch.Tensor(tX[i].copy()).unsqueeze(0).to(device)
        locs, confs = dssd.forward(x)
        locs = decode_offset(locs.squeeze(), default_boxes)
        boxes = non_max_supression(locs, confs) 
        img = tX[i].copy().reshape(512, 512, 3)
        for box in boxes:
            x1 = math.ceil(box[0] * 512)
            y1 = math.ceil(box[1] * 512)
            x2 = math.ceil(box[2] * 512)
            y2 = math.ceil(box[3] * 512)
            start = (x1, y1)
            end = (x2, y2)
            #print(x1,y1, x2, y2)
            img = cv2.rectangle(img, start, end, (0,255,255), 4)
    plot_img(img)


optimizer = torch.optim.SGD(dssd.parameters(), lr=1e-3, momentum=0.9)

epochs = 40
steps_per_epoch = 30
tot_len = X.shape[0]
batch_size = 20
loss = None
classification_loss = 0
regression_loss = 0
best_loss = 1
for j in range(epochs):
    for _ in range(steps_per_epoch):
        mat_boxes = None
        for i,ids in enumerate(get_batch_ids(tot_len, batch_size)):
            x = torch.Tensor(X[ids]).to(device)
            #y_labels = torch.Tensor(Y_labels[ids]).to(device)
            locs, confs = dssd.forward(x)
            classification_loss = 0.0
            regression_loss = 0.0

            for k in range(batch_size):
                ground_boxes = torch.Tensor(Y_annot[ids[k]]).to(device)
                reg_loss, cls_loss, mat_boxes = net_loss(ground_boxes, default_boxes, confs[k], locs[k])
                regression_loss += reg_loss
                classification_loss += cls_loss
            regression_loss /= batch_size
            classification_loss /= batch_size
            loss = regression_loss + classification_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) == batch_size:
                break

        if best_loss > loss:
            best_loss = loss
            torch.save(dssd, 'dssd.pth')

    print(f"Epoch : {j+1} ---------> Loss: {loss}, Classification Loss : {classification_loss.item()}, Regression Loss : {regression_loss.item()}")
    #plot_result(1)

torch.cuda.empty_cache()