# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import scipy.io
from torch.utils.data import DataLoader
from data_loader import MyData, MyTestData


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.c1 = nn.Conv2d(in_channels=1, out_channels=6,
                            kernel_size=5, stride=1, padding=2)
        self.r1 = nn.ReLU(inplace=True)
        self.s2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.c3 = nn.Conv2d(6, 16, 5, 1)
        self.r3 = nn.ReLU(inplace=True)
        self.s4 = nn.MaxPool2d(2, 2)

        # 遇到了卷积层变为全连接层
        self.c5 = nn.Linear(16*5*5, 120)
        self.r5 = nn.ReLU(inplace=True)
        self.f6 = nn.Linear(120, 84)
        self.r6 = nn.ReLU(inplace=True)
        self.f7 = nn.Linear(84, 10)

    def forward(self, x):     # 输入 1*28*28
        out = self.c1(x)      # 6*28*28
        out = self.r1(out)    # 6*28*28
        out = self.s2(out)    # 6*14*14
        out = self.c3(out)    # 16*10*10
        out = self.r3(out)    # 16*10*10
        out = self.s4(out)    # 16*5*5

        out = out.view(-1, 16*5*5)  # out.size()[0], 1*400
        out = self.c5(out)    # 400 --> 120
        out = self.r5(out)
        out = self.f6(out)    # 120 --> 84
        out = self.r6(out)
        out = self.f7(out)    # 84 --> 10
        return out


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 参数设置
# OUTPUT_SIZE = 10
# NUM_EPOCHES = 10
# BATCH_SIZE = 100
# LEARNING_RATE = 0.001

params = {}
params['train_im_path'] = '.\\mnist\\train-images\\'
params['test_im_path'] = '.\\mnist\\t10k-images\\'
params['train_lb_path'] = '.\\mnist\\train-labels.txt'
params['test_lb_path'] = '.\\mnist\\t10k-labels.txt'
params['INPUT_SIZE'] = 784
params['HIDDEN_SIZE'] = 500
params['OUTPUT_SIZE'] = 10
params['NUM_EPOCHES'] = 10
params['BATCH_SIZE'] = 100
params['LEARNING_RATE'] = 0.001

# load dataset
train_loader = DataLoader(MyData(params),
                      shuffle=True,
                      batch_size=params['BATCH_SIZE'])
test_loader = DataLoader(MyTestData(params),
                         shuffle=False,
                         batch_size=1)

model = LeNet().to(device)
model = model.train()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=params['LEARNING_RATE'])

total_step = len(train_loader) #训练数据的大小，也就是含有多少个batch

losses = []
accs = []

for epoch in range(params['NUM_EPOCHES']):
    for i, (images, labels) in enumerate(train_loader):
        # Move tensors to the configured device
        images = images.cuda(device=0)  # # -1 是指模糊控制的意思，即固定784列，不知道多少行
        labels = labels.cuda(device=0)
        # images = images.reshape(-1, 28 * 28).to(device)  # # -1 是指模糊控制的意思，即固定784列，不知道多少行
        # labels = labels.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch+1, params['NUM_EPOCHES'], i+1, total_step, loss.item()))
            losses.append(loss.item())

    # 测试模型
    # 在测试阶段，不用计算梯度
    with torch.no_grad():
        correct = 0
        total = 0
        model = model.eval()
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predict = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predict == labels).sum().item()

        accuracy = correct / total
        accs.append(accuracy)
        print('Accuracy of the network on the 10000 test images: {} %'.format(100 * accuracy))

    model = model.train()

# 保存模型
torch.save(model.state_dict(), 'model_lenet.pth')
scipy.io.savemat('accs_lenet', {'accs': accs})
scipy.io.savemat('loss_lenet', {'loss': losses})
# plot loss figure and accuracy figure
plt.figure()
plt.title('loss')
plt.plot(losses)

plt.figure()
plt.title('accuracy')
plt.plot(accs)

plt.show()















