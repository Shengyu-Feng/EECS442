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
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from preprocess import *
from darknet import *
from preprocess import *
import pickle as pkl
classes = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting','Cyclist','Tram','Misc','DontCare']
class_to_id = dict(zip(classes,range(9)))
id_to_class = dict(zip(range(9),classes))
label_dir = "training/label_2"
train_dir = 'data_object_image_2/training/image_2'
test_dir = 'data_object_image_2/testing/image_2'
colors = pkl.load(open("pallete", "rb"))

def get_data(read_data = [1,1,0], size = 0):
    label = []
    train = []
    test = []
    index = 0
    
    if read_data[0]:
        print('load label...')
        for root, dirs, files in os.walk(label_dir):
            for file in files:
                lines = open(os.path.join(label_dir, file), encoding='utf-8').read().strip().split('\n')
                lines = [line.split() for line in lines]
                lines = [[class_to_id[line[i]] if i==0 else float(line[i]) for i in range(15)] for line in lines]
                lines = np.array(lines)
                label.append(lines[:,np.array([0,4,5,6,7])])
                index += 1
                if size >0 and index >= size:
                    break
    index = 0
    if read_data[1]:
        print('load train...')
        for root, dirs, files, in os.walk(train_dir):
            for file in files:
                img = cv2.imread(os.path.join(train_dir,file))
                train.append(img)
                index += 1
                if size >0 and index >= size:
                    break
    if read_data[2]:
        print('load test')
        for root, dirs, files, in os.walk(test_dir):
            for file in files:
                img = cv2.imread(os.path.join(test_dir,file))
                test.append(img)
    return label, train, test

def write(img, bboxes):
    img_ = img.copy()
    for bbox in bboxes:
        label = "{0}".format(id_to_class[bbox[0]])
        c1 = tuple((bbox[1:3]).astype(np.int32))
        c2 = tuple((bbox[3:5]).astype(np.int32))
        color = random.choice(colors)
        cv2.rectangle(img_, c1, c2, color, 1)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
        c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
        cv2.rectangle(img_, c1, c2,color, -1)
        cv2.putText(img_, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1)
    return img_


# In[2]:


class bboxData(Dataset):
    def __init__(self, size=1000):
        self.size = size
        label, train, test = get_data(size=size)
        image = list(map(letterbox_image, train, [(416,416) for i in range(size)]))
        self.image = torch.Tensor(image).permute(0,3,1,2)
        for i in range(size):
            h, w, c = train[i].shape
            label_ = np.zeros_like(label[i])
            label_[:,1] = label[i][:,1]*416/w
            label_[:,3] = label[i][:,3]*416/w
            label_[:,2] = (label[i][:,2]-h/2)*416/w +416/2
            label_[:,4] = (label[i][:,4]-h/2)*416/w +416/2
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


# In[3]:


model = Darknet('cfg/3dyolo.cfg')
train_data = bboxData(100)
trainloader = DataLoader(train_data, batch_size= 10, shuffle=False)


# In[ ]:


optimizer = optim.Adam(model.parameters(), lr=1e-3)
for epoch in range(20):
    running_loss = 0
    for i, (images, labels) in enumerate(trainloader):
        optimizer.zero_grad()
        loss = model(images, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(running_loss/100)


# In[ ]:




