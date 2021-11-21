import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.optim as optim

import torch.nn as nn
import torch.nn.functional as F
import torch
from models import Discriminator, Generator, weights_init_normal
from utils import inception_eval, inception_eval_cifar10



# beta 1 = 0.0
# xavior 
# reduce lr for D only
# ncritic = 2


class Learner():
    
    
    def __init__(self, opt, z_dim, bs, device):
#     def __init__(self, G, D, G_optimizer, D_optimizer, criterion, z_dim, bs, device):
        
        
        self.opt = opt
        
        
        self.G = Generator(opt).to(device)
        self.criterion = nn.BCELoss().to(device)

        # optimizer
        self.lr = opt.learning_rate   #0.0001
        print(self.lr)
        self.G_optimizer = optim.Adam(self.G.parameters(), lr = self.lr, betas = (0.0, 0.999)) #0.0
        
        
        self.multiD = []
        self.multiD_optim = []
        for d in range(opt.n_dis):
            self.multiD.append(Discriminator(opt).to(device))
            self.multiD_optim.append(optim.Adam(self.multiD[d].parameters(), lr = self.lr/5, betas = (0.0, 0.999)))


        self.device = device
        self.z_dim = opt.z_dim
        self.bs = opt.batch_size

        print(self.multiD[0])
        pytorch_total_params = sum(p.numel() for p in self.multiD[0].parameters())
        print(pytorch_total_params)
        
        
    
    def D_train(self, x_real):
        #=======================Train the discriminator=======================#
        self.G.train()
        for i in range(self.opt.n_dis):
            self.multiD[i].train()
            for p in self.multiD[i].parameters():
                p.requires_grad = True
            self.multiD_optim[i].zero_grad()

        for p in self.G.parameters():
            p.requires_grad = True

        flag = True
        z = Variable(torch.randn(x_real.size()[0], self.opt.z_dim, 1, 1).to(self.device))
        x_fake = self.G(z)
        x_real = Variable(x_real.to(self.device))
        for i in range(self.opt.n_dis):
            if flag:
                D_fake = self.multiD[i](x_fake).unsqueeze(1)
                D_real = self.multiD[i](x_real).unsqueeze(1)
                flag = False
            else:
                D_fake = torch.cat((D_fake, self.multiD[i](x_fake).unsqueeze(1)), dim = 1)
                D_real = torch.cat((D_real, self.multiD[i](x_real).unsqueeze(1)), dim = 1)

        ind = torch.argmin(D_fake, dim = 1)
        mask = torch.zeros((x_real.size()[0], self.opt.n_dis)).to(self.device)

        for i in range(mask.size()[0]):
            random_checker = np.random.randint(0,10)
            if random_checker > 7:
                index = np.random.randint(0,self.opt.n_dis)
                mask[i][index] = 1.0
            else:
                mask[i][ind[i]] = 1.0
        
#         print("XXXXXX", mask.size(), D_fake.size())
        D_fake_output = torch.sum(mask*D_fake.squeeze(2), dim = 1)
        D_real_output = torch.sum(mask*D_real.squeeze(2), dim = 1)
#         print("XXXXXX", mask.size(), D_fake.size())

        y_real = Variable(torch.ones(x_real.size()[0], 1).to(self.device), requires_grad = False)
        y_fake = Variable(torch.zeros(x_real.size()[0], 1).to(self.device), requires_grad = False)

        D_real_loss = self.criterion(D_real_output, y_real)
        D_fake_loss = self.criterion(D_fake_output, y_fake)

        D_acc = self.get_critic_acc(D_fake_output, D_real_output)

        # gradient backprop & optimize ONLY D's parameters
        D_loss = D_real_loss + D_fake_loss
        D_loss.backward()

        for i in range(self.opt.n_dis):
            if i in ind:
                self.multiD_optim[i].step()
        # multiD_optim[dis_index].step()

        return  D_loss.data.item(), D_acc, ind

    def G_train(self, x):
        #=======================Train the generator=======================#
        self.G.train()
        self.G_optimizer.zero_grad()

        z = Variable(torch.randn(self.opt.batch_size, self.opt.z_dim, 1, 1).to(self.device))
        y = Variable(torch.ones(self.opt.batch_size, 1).to(self.device), requires_grad = False)

        critic_fakes = []
        fake_img = self.G(z)
        lit = np.zeros(self.opt.n_dis)
        for i in range(self.opt.n_dis):
            for p in self.multiD[i].parameters():
                p.requires_grad = False
            critic_fake = self.multiD[i](fake_img)
            critic_fakes.append(critic_fake)
            lit[i] = torch.sum(critic_fake).item()
        loss_sort = np.argsort(lit)
        weights = np.random.dirichlet(np.ones(self.opt.n_dis))
        weights = np.sort(weights)[::-1]
        # weights = np.array([1.0])
        # weights[0] = weights[0]+weights[n_dis-1]
        # weights[n_dis-1] = 0.0

        flag = False
        for i in range(len(critic_fakes)):
            if flag == False:
                critic_fake = weights[i]*critic_fakes[loss_sort[i]]
                flag = True
            else:
                critic_fake = torch.add(critic_fake, weights[i]*critic_fakes[loss_sort[i]])

        # critic_fake = multiD[0](fake_img)
        G_loss = self.criterion(critic_fake, y)

        # gradient backprop & optimize ONLY G's parameters
        G_loss.backward()
        self.G_optimizer.step()

        return G_loss.data.item()

    def get_critic_acc(self, critic_fake, critic_real):
        acc = 0.0
        for x in critic_fake[:]:
            if x.item() <= 0.5:
                acc = acc + 1
        for x in critic_real[:]:
            if x.item() >= 0.5:
                acc = acc + 1

        acc = acc/(critic_fake.size()[0] + critic_real.size()[0])
        return acc
    
    def adjust_learning_rate(self, factor):
    
        new_lr = self.lr
        # for param_group in self.G_optimizer.param_groups:
        #     param_group['lr'] = new_lr
        # for param_group in self.D_optimizer.param_groups:
        #     param_group['lr'] = new_lr