# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 07:23:12 2021
0
@author: 严天宇
"""

import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import math
import copy

EPOCH = 1
BATCH_SIZE = 50
LR = 0.001
DOWNLOAD_MNIST = False

train_data = torchvision.datasets.MNIST(
        root = './mnist/',
        train = True,
        transform = torchvision.transforms.ToTensor(),
        download = DOWNLOAD_MNIST,
        )
#torch_dataset = Data.TensorDataset(x, y), 建立自己的数据库

print(train_data.train_data.size())
print(train_data.train_labels.size())

train_loader = Data.DataLoader(dataset = train_data,batch_size =BATCH_SIZE,shuffle = True)

test_data = torchvision.datasets.MNIST(root = './mnist/',train = False)
test_x = torch.unsqueeze(test_data.test_data,dim=1).type(torch.FloatTensor)[:2000]/255.
test_y = test_data.test_labels[:2000]


class Attention(nn.Module):
    def __init__(self):   
        #config.hidden_size=768
        #config.transformer.num_heads = 12
        super(Attention, self).__init__()
        self.num_attention_heads = 8
        self.attention_head_size = 4
        self.all_head_size = 32

        self.query = nn.Linear(32, 32)
        self.key = nn.Linear(32, 32)
        self.value = nn.Linear(32, 32)

        self.out = nn.Linear(32, 32)

        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0,2, 1, 3)  #batch_size,num_head, H*W, head_size

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)  #相当于把原始乱七八糟的通道数，改成适合分组的通道数
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))#各个列vector注意力分配
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)#对dim = -1的理解在于，所有的数值都在最后一维


        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0,2, 1, 3).contiguous()    #batch_size,H*W,num_heads,head_size
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)#batch_size,H*W,C
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        return attention_output
    
class Mlp(nn.Module):
    def __init__(self):
        super(Mlp, self).__init__()
        #config.transformer.mlp_dim = 3072[]
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, 32)
        self.act_fn = nn.ReLU()
        #self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        #x = self.dropout(x)
        x = self.fc2(x)
        #x = self.dropout(x)
        return x
    
class Block(nn.Module):
    def __init__(self):
        super(Block, self).__init__()
        self.attention_norm = nn.LayerNorm(32, eps=1e-6)
        self.ffn_norm = nn.LayerNorm(32, eps=1e-6)
        self.ffn = Mlp()
        self.attn = Attention()

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x


#Encoder是不停的堆叠transformer
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.layer = nn.ModuleList()  #类似python里的列表
        self.encoder_norm = nn.LayerNorm(32, eps=1e-6)
        for _ in range(3):
            layer = Block()
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        for layer_block in self.layer:
            #也就是hidden_states的尺寸一直没有变，只是加深理解而已
            hidden_states = layer_block(hidden_states)  #hidden_states可以一直传下去
        encoded = self.encoder_norm(hidden_states)
        return encoded    



class cnnn(nn.Module):
    def __init__(self):
        super(cnnn,self).__init__()
        self.conv1 = nn.Sequential(
                nn.Conv2d(1,16,5,1,2),
                nn.ReLU(),
                nn.MaxPool2d(2),
                )
        self.conv2 = nn.Sequential(
                nn.Conv2d(16,32,5,1,2),
                nn.ReLU(),
                nn.MaxPool2d(2),      
                )           #32*7*7
        self.dropout = nn.Dropout()
        self.out = nn.Linear(32*7*7, 10)    #conv和linear都可以对带batch_size的数据进行操作，不变batch_size
        
    def forward(self,x):
        x = self.conv1(x)
        #x = self.dropout(x)
        x = self.conv2(x)
        #x = self.dropout(x)
        #x = x.view(x.size(0),-1)      #一定要展平！
        #x = self.out(x)
        
        return x

class Encoder_func(nn.Module):
    def __init__(self):
        super(Encoder_func,self).__init__()
        self.encoder = Encoder()
        self.cnn = cnnn()
        self.strange = nn.Linear(32*49,10)

    def forward(self,x):
        x = self.cnn(x)
        x = x.permute(0,2,3,1)
        x = x.view(x.size(0),7*7,32)
        output = self.encoder(x)
        output = output.view(output.size(0),-1)
        output = self.strange(output)
        return output

#cnn = cnnn()
#print(cnn)
trans = Encoder_func()
optimizer = torch.optim.Adam(trans.parameters(),lr = LR)
loss_func = nn. CrossEntropyLoss()
strange = nn.Linear(32*49,10)

for epoch in range(EPOCH):
    for step,(b_x,b_y) in enumerate(train_loader):#这里的b_x是batch*c*h*w
        #output = cnn(b_x)
        output = trans(b_x)
        #print(b_x.size())
        #output = trans(b_x)  #50,49,32
        #output = output.view(output.size(0),-1)
        #output = strange(output)
        loss = loss_func(output,b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 50 == 0:
            #test_output = cnn(test_x)
            test_output = trans(test_x)
            pred_y = torch.max(test_output, 1)[1].data.squeeze().numpy()
            accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)
        
       
        
    