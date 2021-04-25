# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 21:36:51 2021

@author: 严天宇
"""

import torch
import torch.nn as nn
import os
from PIL import Image
from torchvision import transforms
import numpy as np
from torch.utils.data import Dataset, DataLoader
 
filePath = 'train\\train'
name = os.listdir(filePath)
test_name1 = name[:2500]
test_name2 = name[22500:]
name = name[2500:]
name = name[:20000]

test_name = test_name1+test_name2

params = {}
params['filePath'] = filePath
params['BATCH_SIZE'] = 50
params['NUM_EPOCHES'] = 1
params['LEARNING_RATE'] = 0.001


def transform(img):
        # flip, crop, rotate
        p = np.random.rand(1)
        if p >= 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        # rotate
        angle = np.random.uniform(-10, 10)
        img.rotate(angle)

        transform_img = transforms.Compose([transforms.Resize((512, 512)),
                                            transforms.ToTensor()])
        im = transform_img(img)
        return im/255.

test_img = []
test_label = []
for i in range(5000):
    img_file = os.path.join(filePath,test_name[i])
    img = Image.open(img_file)
    im = transform(img)
    test_img.append(im)

    lb = str(test_name[i])
    if lb[:3]=='cat':
        lb = 0
    else:
        lb =1
    test_label.append(lb)

test_label = torch.tensor(test_label)
    
class MyData(Dataset):
    def __init__(self,params):
        self.filePath = params['filePath']
        self.name = name
        self.train_num = 20000
        
    def __len__(self):
        return self.train_num
    
    def __getitem__(self,index):
        img_file = os.path.join(self.filePath,self.name[index])
        img = Image.open(img_file)
        im = self.transform(img)
        
        # load label
        lb = str(self.name[index])
        if lb[:3]=='cat':
            lb = 0
        else:
            lb =1

        return im, lb
        
    def transform(self,img):
        # flip, crop, rotate
        p = np.random.rand(1)
        if p >= 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        # rotate
        angle = np.random.uniform(-10, 10)
        img.rotate(angle)

        transform_img = transforms.Compose([transforms.Resize((512, 512)),
                                            transforms.ToTensor()])
        im = transform_img(img)
        return im/255.
    #img = io.imread(filePath + "\\" + i)
'''   
class test_data(Dataset):
    def __init__(self,params):
        self.filePath = params['filePath']
        self.test_name = test_name
        self.test_num = 5000
        
    def __len__(self):
        return self.train_num
    
    def __getitem__(self,index):
        img_file = os.path.join(self.filePath,self.test_name[index])
        img = Image.open(img_file)
        im = self.transform(img)
        
        # load label
        lb = str(self.name[index])
        if lb[:3]=='cat':
            lb = 0
        else:
            lb =1

        return im, lb
    
    def transform(self,img):
        transform_img = transforms.Compose([transforms.Resize((512, 512)),
                                            transforms.ToTensor()])
        im = transform_img(img)
        return im/255.
'''   
class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)

class Encoder(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.pool = nn.AvgPool2d(2)
        #只用decoderblock有升维的函数

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool(x)
        return x
    
class Mlp(nn.Module):
    def __init__(self):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(256*16*16, 256)
        self.fc2 = nn.Linear(256, 2)
        self.act_fn = nn.ReLU()
        self.dropout = nn.Dropout()

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
    
class VGG16(nn.Module):
    def __init__(self):
        super().__init__()
        self.encode1 = Encoder(3,16) #256
        self.encode2 = Encoder(16,32)  #128
        self.encode3 = Encoder(32,64)    #64
        self.encode4 = Encoder(64,128)   #32
        self.encode5 = Encoder(128,256)    #16
        self.mlp = Mlp()
    
    def forward(self,x):
        x = self.encode1(x)
        x = self.encode2(x)
        x = self.encode3(x)
        x = self.encode4(x)
        x = self.encode5(x)
        x = x.view(x.size(0),-1)
        x = self.mlp(x)
        
        return x


train_loader = DataLoader(MyData(params),
                      shuffle=True,
                      batch_size=params['BATCH_SIZE'])
'''test_loader = DataLoader(test_data(params),
                         shuffle=True,
                         batch_size = 1)
'''

model = VGG16().cuda()
model = model.train()

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=params['LEARNING_RATE'])

for epoch in range(params['NUM_EPOCHES']):
    for i, (images, labels) in enumerate(train_loader):
        images = images.cuda()
        labels = labels.cuda()
        
        outputs = model(images)
        loss = loss(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if i % 50 == 0:
            test_output = model(test_img)
            pred_y = torch.max(test_output, 1)[1].data.squeeze().numpy()
            accuracy = float((pred_y == test_label.data.numpy()).astype(int).sum()) / float(test_label.size(0))
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.4f' % accuracy)
