# -*- coding: utf-8 -*-
"""
Created on Sun May 30 20:48:33 2021

@author: 严天宇
"""
from unet_vgg import U_Net

import torchvision.models as models


net = U_Net()

model = models.vgg16(pretrained = True)

pretrained_dict = model.state_dict()


model_dict = net.state_dict()  #返

# 1. filter out unnecessary keys，也就是说从内置模块中删除掉我们不需要的字典
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
print(pretrained_dict.keys())
 
# 2. overwrite entries in the existing state dict，利用pretrained_dict更新现有的model_dict
model_dict.update(pretrained_dict)
'''
#查看参数名称和数值
for name, parameters in model.named_parameters():
    print(name, ':', parameters.size())
'''
# 3. load the new state dict，更新模型，加载我们真正需要的state_dict

net.load_state_dict(model_dict)



#torch.nn.init.kaiming_normal_(tensor, a=0, mode='fan_in')
#这是额外添加的矩阵，赋值为空