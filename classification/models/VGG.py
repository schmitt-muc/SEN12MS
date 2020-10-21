# Modified from Jian Kang, https://www.rsim.tu-berlin.de/menue/team/dring_jian_kang/
# Modified by Yu-Lun Wu, TUM
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Function
from torchvision import models


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def weights_init_kaiming(m):  # initialize the weights (kaiming method)
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.kaiming_normal_(m.weight.data)

def fc_init_weights(m):
    if type(m) == nn.Linear:
        init.kaiming_normal_(m.weight.data)

    
class VGG16(nn.Module):
    def __init__(self, n_inputs = 12, numCls = 17):  # num. of classes
        super().__init__()

        vgg = models.vgg16(pretrained=False)

        self.encoder = nn.Sequential(
            nn.Conv2d(n_inputs, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),  #  12 bands as input (s1 + s2)
            *vgg.features[1:]
        )
        self.classifier = nn.Sequential(
            nn.Linear(8*8*512, 4096, bias=True),   # 8*8*512: output size from encoder (origin img pixel 256*256-> 5 pooling = 8)
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, numCls, bias=True)
        )

        self.apply(weights_init_kaiming)
        self.apply(fc_init_weights)

    def forward(self, x):

        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        logits = self.classifier(x)

        return logits 


class VGG19(nn.Module):
    def __init__(self, n_inputs = 12, numCls = 17): #
        super().__init__()

        vgg = models.vgg19(pretrained=False)

        self.encoder = nn.Sequential(
            nn.Conv2d(n_inputs, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), #
            *vgg.features[1:]
        )
        self.classifier = nn.Sequential(
            nn.Linear(8*8*512, 4096, bias=True), #
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, numCls, bias=True)
        )

        self.apply(weights_init_kaiming)
        self.apply(fc_init_weights)

    def forward(self, x):

        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        logits = self.classifier(x)

        return logits 




if __name__ == "__main__":
    
    #n_inputs = 2

    inputs = torch.randn((1, 12, 256, 256)) # (how many images, spectral channels, pxl, pxl)
    #net = VGG16(n_inputs)
    net = VGG16()
    #net = VGG19()
    #numParams = count_parameters(net)
    outputs = net(inputs)
    print(outputs)


