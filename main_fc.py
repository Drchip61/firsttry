# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data_loader import MyData, MyTestData
import matplotlib.pyplot as plt
import pdb


class FCNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FCNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 参数设置
params = {}
# params['train_im_path'] = '.\\mnist\\train-images\\'
# params['test_im_path'] = '.\\mnist\\t10k-images\\'
# params['train_lb_path'] = '.\\mnist\\train-labels.txt'
# params['test_lb_path'] = '.\\mnist\\t10k-labels.txt'


params['train_im_path'] = 'mnist\\train-images\\'
params['test_im_path'] = 'mnist\\t10k-images\\'
params['train_lb_path'] = 'mnist\\train-labels.txt'
params['test_lb_path'] = 'mnist\\t10k-labels.txt'

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

model = FCNet(params['INPUT_SIZE'], params['HIDDEN_SIZE'], params['OUTPUT_SIZE']).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=params['LEARNING_RATE'])

total_step = len(train_loader) #训练数据的大小，也就是含有多少个batch

losses = []
accs = []

for epoch in range(params['NUM_EPOCHES']):
    for i, (images, labels) in enumerate(train_loader):
        # Move tensors to the configured device
        # images = images.reshape(-1, 28*28).cuda(device=0)  # # -1 是指模糊控制的意思，即固定784列，不知道多少行
        # labels = labels.cuda(device=0)
        images = images.reshape(-1, 28 * 28).to(device)  # # -1 是指模糊控制的意思，即固定784列，不知道多少行
        labels = labels.to(device)

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
            images = images.reshape(-1, 28 * 28).to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predict = torch.max(outputs.data, 1)  
            ##这里返回两组数据，最大image_data和最大值索引，可以用torch.argmax（）更为直观；这里去理解其作用为返回最大索引，即预测出来的类别。
            ## 这个 _ , predicted是python的一种常用的写法，表示后面的函数其实会返回两个值
            # 但是我们对第一个值不感兴趣，就写个_在那里，把它赋值给_就好，我们只关心第二个值predicted
            # 比如 _ ,a = 1,2 这中赋值语句在python中是可以通过的，你只关心后面的等式中的第二个位置的值是多少
            total += labels.size(0)  ##更新测试图片的数量   size(0),返回行数
            correct += (predict == labels).sum().item()

        accuracy = correct / total
        accs.append(accuracy)
        print('Accuracy of the network on the 10000 test images: {} %'.format(100 * accuracy))

    model = model.train()

# plot loss figure and accuracy figure
plt.figure()
plt.title('loss')
plt.plot(losses)

plt.figure()
plt.title('accuracy')
plt.plot(accs)

plt.show()

# 保存模型
torch.save(model.state_dict(), 'model.pth')










