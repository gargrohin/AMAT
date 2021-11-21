import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch


def weights_init_normal(m):
    classname = m.__class__.__name__
    print(classname)
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
#         torch.nn.init.xavier_uniform(m.weight.data)  
    elif classname.find("ConvTranspose2d") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)



class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        
        self.img_size = 64
        self.latent_dim = opt.z_dim
        self.channels = 3
        
        a = 128
        k = 5
        P = 2
        m = 0.8
        self.a = a
        
        self.init_size = self.img_size // 16
#         print("XXXXXX", self.init_size)
        self.l1 = nn.Sequential(nn.Linear(self.latent_dim, a*8 * self.init_size ** 2, bias=False))
        
        
        
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(a*8),
            nn.Upsample(scale_factor=2),
            
            nn.Conv2d(a*8, a*4, k, stride=1, padding=P, bias=False),
            
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(a*4, m),
            nn.Upsample(scale_factor=2),
            
            nn.Conv2d(a*4, a*2, k, stride=1, padding=P, bias=False),
            
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(a*2,  m),
            nn.Upsample(scale_factor=2),
            
            nn.Conv2d(a*2, a, k, stride=1, padding=P, bias=False),
            
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(a,  m),
            nn.Upsample(scale_factor=2),
            
            nn.Conv2d(a, self.channels, k, stride=1, padding=P, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
#         print(z.size(), self.init_size)
        z = z.view(z.size(0), self.latent_dim)
        out = self.l1(z)
        out = out.view(out.shape[0], self.a*8, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()

        a = 128
        self.channels = 3
        self.img_size = 64
        
        
        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 5, 2, 2, bias=False), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.0)]
            if bn:
#                 block = [nn.Conv2d(in_filters, out_filters, 5, 2, 2, bias=False), nn.BatchNorm2d(out_filters,  0.8), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.0)]
#                 return block
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block
        
        
        self.model = nn.Sequential(
            *discriminator_block(self.channels, a, bn=False),
            *discriminator_block(a, a*2),
            *discriminator_block(a*2, a*4),
            *discriminator_block(a*4, a*8),
        )
    
        # The height and width of downsampled image
        ds_size = self.img_size // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(a*8 * ds_size ** 2, 1, bias=False), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity

    