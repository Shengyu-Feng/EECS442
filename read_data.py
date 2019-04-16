#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os 
import string
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch.autograd import Variable
import random
from torch import optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from preprocess import *
from darknet import *
from util import *
import pickle as pkl
batch_size = 20
classes = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting','Cyclist','Tram','Misc','DontCare']
class_to_id = dict(zip(classes,range(9)))
id_to_class = dict(zip(range(9),classes))
label_dir = "training/label_2"
train_dir = 'training/image_2'
test_dir = 'testing/image_2'
colors = pkl.load(open("pallete", "rb"))
device = "cuda" if torch.cuda.is_available() else "cpu"
def get_data(read_data = [1,1,0], size = [7481,7481,7518]):
    label = []
    train = []
    test = []
    index = 0
    
    if read_data[0]:
        print('load label...')
        for root, dirs, files in os.walk(label_dir):
            for file in sorted(files):
                lines = open(os.path.join(label_dir, file), encoding='utf-8').read().strip().split('\n')
                lines = [line.split() for line in lines]
                lines = [[class_to_id[line[i]] if i==0 else float(line[i]) for i in range(15)] for line in lines]
                lines = np.array(lines)
                label.append(lines[:,np.array([0,4,5,6,7])])
                #label.append(lines)
                index += 1
                if index >= size[0]:
                    break
    index = 0
    if read_data[1]:
        print('load train...')
        for root, dirs, files, in os.walk(train_dir):
            for file in sorted(files):
                img = cv2.imread(os.path.join(train_dir,file))
                train.append(img)
                index += 1
                if index >= size[1]:
                    break
    index = 0
    if read_data[2]:
        print('load test')
        for root, dirs, files, in os.walk(test_dir):
            for file in files:
                img = cv2.imread(os.path.join(test_dir,file))
                test.append(img)
                index += 1
                if index>=size[2]:
                    break
    return label, train, test

def write(img, bboxes):
    img_ = img.copy()
    for bbox in bboxes:
        label = "{0}".format(id_to_class[bbox[0].astype(int)])
        c1 = tuple((bbox[1:3]).astype(np.int32))
        c2 = tuple((bbox[3:5]).astype(np.int32))
        color = random.choice(colors)
        cv2.rectangle(img_, c1, c2, color, 1)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
        c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
        cv2.rectangle(img_, c1, c2,color, -1)
        cv2.putText(img_, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1)
    return img_

class bboxData(Dataset):
    def __init__(self, size=1000):
        self.size = size
        label, train, test = get_data(size=[size,size,0])
        image = list(map(cv2.resize, train, [(416,416) for i in range(size)]))
        self.image = torch.FloatTensor(image).permute(0,3,1,2).div(255.0)
        for i in range(size):
            h, w, c = train[i].shape
            label_ = np.zeros_like(label[i])
            label_[:,1] = label[i][:,1]*416/w
            label_[:,3] = label[i][:,3]*416/w
            label_[:,2] = label[i][:,2]*416/h
            label_[:,4] = label[i][:,4]*416/h
            label[i][:,1] = (label_[:,3] + label_[:,1])/832
            label[i][:,2] = (label_[:,4] + label_[:,2])/832
            label[i][:,3] = (label_[:,3] - label_[:,1])/416
            label[i][:,4] = (label_[:,4] - label_[:,2])/416
            label[i] = torch.Tensor(label[i])
        self.label = pad_sequence(label,True)     
        
    def __len__(self):
        return self.size
    
    def __getitem__(self,idx):
        sample = (self.image[idx],self.label[idx])
        return sample


# In[ ]:


train_data = bboxData(size = 7481)
model = Darknet('cfg/yolov3.cfg')
model.load_weights("yolov3.weights")
model.module_list[81] = nn.Conv2d(1024,42,1,1)
model.blocks[83]['classes'] = '9'
model.module_list[93] = nn.Conv2d(512, 42,1,1)
model.blocks[95]['classes'] = '9'
model.module_list[105] = nn.Conv2d(256, 42,1,1)
model.blocks[107]['classes'] = '9'
"""
model = Darknet('cfg/3dyolo.cfg')
weights = torch.load('model_3dyolo.pth', map_location='cpu')
model.load_state_dict(weights)
"""


# In[ ]:


def train(model, trainloader, optimizer, device):
    running_loss = 0
    model.train()
    for i, (images, labels) in enumerate(trainloader):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        loss = model(images, labels,True)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if (i +1) % 10 ==0:
            print("Iteration %d: Training loss: %f"%(i+1, running_loss/10))
            running_loss = 0

def validate(model, valloader, device):
    running_loss = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for i, (images, labels) in enumerate(valloader):
            images = images.to(device)
            labels = labels.to(device)
            loss = model(images, labels, True)
            running_loss += loss.item()
            total += 1
        print("Validation loss: %f"%(running_loss/total))

train_size = int(0.9*len(train_data))
val_size = len(train_data) - train_size
trainset, valset = random_split(train_data, [train_size,val_size])
trainloader = DataLoader(trainset, batch_size= batch_size, shuffle=False)
valloader = DataLoader(valset, batch_size=1,shuffle=False)


# In[ ]:


optimizer = optim.Adam(model.parameters(), lr=1e-5)
epoches = 2
for epoch in range(epoches):
    train(model, trainloader, optimizer, device)
    validate(model,valloader, device)


# In[ ]:


torch.save(model.state_dict(), './model_3dyolo.pth')

