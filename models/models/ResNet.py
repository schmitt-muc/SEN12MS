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


class ResNet18(nn.Module):
    def __init__(self, n_inputs = 12, numCls = 17):
        super().__init__()

        resnet = models.resnet18(pretrained=False)
        
        self.conv1 = nn.Conv2d(n_inputs, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.encoder = nn.Sequential(
            self.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool
        )
        self.FC = nn.Linear(512, numCls)

        self.apply(weights_init_kaiming)
        self.apply(fc_init_weights)

    def forward(self, x):

        x = self.encoder(x)
        x = x.view(x.size(0), -1)

        logits = self.FC(x)

        return logits




class ResNet34(nn.Module):
    def __init__(self, n_inputs = 12, numCls = 17):
        super().__init__()

        resnet = models.resnet34(pretrained=False)

        self.conv1 = nn.Conv2d(n_inputs, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.encoder = nn.Sequential(
            self.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool
        )
        self.FC = nn.Linear(512, numCls)
        
        self.apply(weights_init_kaiming)
        self.apply(fc_init_weights)

    def forward(self, x):

        x = self.encoder(x)
        x = x.view(x.size(0), -1)

        logits = self.FC(x)

        return logits



class ResNet50(nn.Module):
    def __init__(self, n_inputs = 12, numCls = 17):
        super().__init__()

        resnet = models.resnet50(pretrained=False)

        self.conv1 = nn.Conv2d(n_inputs, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.encoder = nn.Sequential(
            self.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool
        )
        self.FC = nn.Linear(2048, numCls)

        self.apply(weights_init_kaiming)
        self.apply(fc_init_weights)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)

        logits = self.FC(x)

        return logits

#class ResNet50_em512(nn.Module):
#    def __init__(self, n_inputs = 12, numCls = 17):
#        super().__init__()
#
#        resnet = models.resnet50(pretrained=False)
#
#        self.conv1 = nn.Conv2d(n_inputs, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
#        self.encoder = nn.Sequential(
#            self.conv1,
#            resnet.bn1,
#            resnet.relu,
#            resnet.maxpool,
#            resnet.layer1,
#            resnet.layer2,
#            resnet.layer3,
#            resnet.layer4,
#            resnet.avgpool
#        )
#        self.FC1 = nn.Linear(2048, 512)
#        self.FC2 = nn.Linear(512, numCls)
#
#        self.apply(weights_init_kaiming)
#        self.apply(fc_init_weights)
#
#    def forward(self, x):
#        x = self.encoder(x)
#        x = x.view(x.size(0), -1)
#
#        x = self.FC1(x)
#        logits = self.FC2(x)
#
#        return logits
#
#
#class ResNet50_em(nn.Module):
#    def __init__(self, n_inputs = 12, numCls = 17):
#        super().__init__()
#
#        resnet = models.resnet50(pretrained=False)
#
#        self.conv1 = nn.Conv2d(n_inputs, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
#        self.encoder = nn.Sequential(
#            self.conv1,
#            resnet.bn1,
#            resnet.relu,
#            resnet.maxpool,
#            resnet.layer1,
#            resnet.layer2,
#            resnet.layer3,
#            resnet.layer4,
#            resnet.avgpool
#        )
#        self.FC = nn.Linear(2048, numCls)
#
#        self.apply(weights_init_kaiming)
#        self.apply(fc_init_weights)
#
#    def forward(self, x):
#        x = self.encoder(x)
#        x = x.view(x.size(0), -1)
#
#        logits = self.FC(x)
#
#        return logits, x

class ResNet101(nn.Module):
    def __init__(self, n_inputs = 12, numCls = 17):
        super().__init__()

        resnet = models.resnet101(pretrained=False)

        self.conv1 = nn.Conv2d(n_inputs, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.encoder = nn.Sequential(
            self.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool
        )
        self.FC = nn.Linear(2048, numCls)
        
        self.apply(weights_init_kaiming)
        self.apply(fc_init_weights)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)

        logits = self.FC(x)

        return logits



class ResNet152(nn.Module):
    def __init__(self, n_inputs = 12, numCls = 17):
        super().__init__()

        resnet = models.resnet152(pretrained=False)

        self.conv1 = nn.Conv2d(n_inputs, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.encoder = nn.Sequential(
            self.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool
        )
        self.FC = nn.Linear(2048, numCls)

        self.apply(weights_init_kaiming)
        self.apply(fc_init_weights)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)

        logits = self.FC(x)

        return logits


if __name__ == "__main__":
    
    inputs = torch.randn((1, 12, 256, 256)) # (how many images, spectral channels, pxl, pxl)

    net = ResNet18()
    #net = ResNet34()
    #net = ResNet50()
    #net = ResNet101()
    #net = ResNet152()

    outputs = net(inputs)

    print(outputs)
    print(outputs.shape)

    numParams = count_parameters(net)

    print(f"{numParams:.2E}")


