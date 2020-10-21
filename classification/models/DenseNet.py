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

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.kaiming_normal_(m.weight.data)

def fc_init_weights(m):
    if type(m) == nn.Linear:
        init.kaiming_normal_(m.weight.data)


class DenseNet121(nn.Module):
    def __init__(self, n_inputs = 12, numCls = 17):
        super().__init__()

#        densenet = models.densenet121(pretrained=False, memory_efficient=True)
        densenet = models.densenet121(pretrained=False)
        
        self.encoder = nn.Sequential(
                nn.Conv2d(n_inputs, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
                *densenet.features[1:])
        
        # classifier
        self.classifier = nn.Linear(65536, numCls, bias=True)

        self.apply(weights_init_kaiming)
        self.apply(fc_init_weights)
        
        
    def forward(self, x):

        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        logits = self.classifier(x)

        return logits
    
    
class DenseNet161(nn.Module):
    def __init__(self, n_inputs = 12, numCls = 17):
        super().__init__()

        densenet = models.densenet161(pretrained=False)
        
        self.encoder = nn.Sequential(
                nn.Conv2d(n_inputs, 96, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
#                nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                *densenet.features[1:])
        
        # classifier
        self.classifier = nn.Linear(141312, numCls, bias=True)

        self.apply(weights_init_kaiming)
        self.apply(fc_init_weights)
        
        
    def forward(self, x):

        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        logits = self.classifier(x)

        return logits 
    
    
    
class DenseNet169(nn.Module):
    def __init__(self, n_inputs = 12, numCls = 17):
        super().__init__()

        densenet = models.densenet169(pretrained=False)
        
        self.encoder = nn.Sequential(
                nn.Conv2d(n_inputs, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
                *densenet.features[1:])
        
        # classifier
        self.classifier = nn.Linear(106496, numCls, bias=True)

        self.apply(weights_init_kaiming)
        self.apply(fc_init_weights)
        
        
    def forward(self, x):

        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        logits = self.classifier(x)

        return logits 
    
    
class DenseNet201(nn.Module):
    def __init__(self, n_inputs = 12, numCls = 17):
        super().__init__()

        densenet = models.densenet201(pretrained=False)
        
        self.encoder = nn.Sequential(
                nn.Conv2d(n_inputs, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
                *densenet.features[1:])
        
        # classifier
        self.classifier = nn.Linear(122880, numCls, bias=True)

        self.apply(weights_init_kaiming)
        self.apply(fc_init_weights)
        
        
    def forward(self, x):

        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        logits = self.classifier(x)

        return logits    
    
    
    
    

if __name__ == "__main__":
    
    inputs = torch.randn((2, 12, 256, 256)) # (how many images, spectral channels, pxl, pxl)
    
    #
    import time
    start_time = time.time()
    #
    
    net = DenseNet121()


    outputs = net(inputs)
    
    print("--- %s seconds ---" % (time.time() - start_time))

    print(outputs)
    print(outputs.shape)

    numParams = count_parameters(net)

    print(f"{numParams:.2E}")
 


  