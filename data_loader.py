import scipy.io as io
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
import pdb
import os
from PIL import Image
import numpy as np

class MyData(Dataset):
    def __init__(self, params):
        super(MyData, self).__init__()
        self.train_im_path = params['train_im_path']
        self.train_lb_path = params['train_lb_path']
        self.train_im_num = 60000
        self.train_labels = open(self.train_lb_path, 'r').read().splitlines()

    def __len__(self):
        return self.train_im_num

    def __getitem__(self, index):
        # load image
        img_file = os.path.join(self.train_im_path, str(index)+'.png')
        img = Image.open(img_file)
        im = self.transform(img)

        # load label
        lb = int(self.train_labels[index])

        return im, lb

    def transform(self, img):
        # flip, crop, rotate
        p = np.random.rand(1)
        if p >= 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        # rotate
        angle = np.random.uniform(-10, 10)
        img.rotate(angle)

        transform_img = transforms.Compose([transforms.Resize((28, 28)),
                                            transforms.ToTensor()])
        im = transform_img(img)
        return im


class MyTestData(Dataset):
    def __init__(self, params):
        super(MyTestData, self).__init__()
        self.test_im_path = params['test_im_path']
        self.test_lb_path = params['test_lb_path']
        self.test_im_num = 10000
        self.test_labels = open(self.test_lb_path, 'r').read().splitlines()

    def __len__(self):
        return self.test_im_num

    def __getitem__(self, index):
        # load image
        img_file = os.path.join(self.test_im_path, str(index) + '.png')
        img = Image.open(img_file)
        im = self.transform(img)

        # load label
        lb = int(self.test_labels[index])

        return im, lb

    def transform(self, img):
        transform_img = transforms.Compose([transforms.Resize((28, 28)),
                                            transforms.ToTensor()])
        im = transform_img(img)
        return im


