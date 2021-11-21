import comet_ml
import logging
comet_ml.config.save(api_key="CX4nLhknze90b8yiN2WMZs9Vw")
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("comet_ml")

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="5"

# prerequisites
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image
import numpy as np
import os
import torchvision.utils as vutils

import os.path
import sys
import tarfile

from six.moves import urllib
import tensorflow as tf
import glob
import scipy.misc
import math
import sys

from scipy.stats import entropy

from inception import get_inception_score

transform = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

# def load_data_STL10():
#     train_dataset =dsets.STL10(root='./data/', split='train+unlabeled', download=True, transform=transform)
#     train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
#     return train_loader

def load_data_CIFAR10():
    train_dataset = datasets.CIFAR10(root='../datasets/cifar10_data/', train=True,download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    return train_loader

class Log_loss(torch.nn.Module):
    def __init__(self):
        # negation is true when you minimize -log(val)
        super(Log_loss, self).__init__()
       
    def forward(self, x, negation=True):
        # shape of x will be [batch size]
        log_val = torch.log(x)
        loss = torch.sum(log_val)
        if negation:
            loss = torch.neg(loss)
        return loss
    
class Itself_loss(torch.nn.Module):
    def __init__(self):
        super(Itself_loss, self).__init__()
        
    def forward(self, x, negation=True):
        # shape of x will be [batch size]
        loss = torch.sum(x)
        if negation:
            loss = torch.neg(loss)
        return loss

nz = 100
nc = 3
ngf = 64
ndf = 64

class _netG(nn.Module):
    def __init__(self):
        super(_netG, self).__init__()
        self.main = nn.Sequential(
            
            # Z
            nn.ConvTranspose2d(nz, ngf*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            
            # (ngf * 8) x 2 x 2
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            
            # (ngf * 4) x 4 x 4
            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),
            
            # (ngf * 2) x 8 x 8
            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(),
            
            # ngf x 16 x 16
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    def forward(self, input):
        output = self.main(input)
        return output

class _netD(nn.Module):
    def __init__(self):
        super(_netD, self).__init__()
        self.main = nn.Sequential(
            # (nc) x 32 x 32
            nn.Conv2d(nc, ndf, 4,2,1,bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # ndf x 16 x 16
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # (ndf * 2) x 8 x 8
            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # (ndf * 4) x 4 x 4
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2, inplace=True),

            # (ndf * 8) x 2 x 2
            nn.Conv2d(ndf*8, 1, 4, 1, 0, bias=False),
            nn.Softplus()
        )
    def forward(self, input):
        output = self.main(input)
        return output.view(-1, 1).squeeze(1)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:         # Conv weight init
        m.weight.data.normal_(0.0, 0.01)
    elif classname.find('BatchNorm') != -1:  # BatchNorm weight init
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def get_netG():
    use_cuda = torch.cuda.is_available()
    netG = _netG()
    netG.apply(weights_init)
    if use_cuda:
        print("USE CUDA")
        netG.cuda()
    return netG

def get_netD():
    use_cuda = torch.cuda.is_available()
    netD = _netD()
    netD.apply(weights_init)
    if use_cuda:
        print("USE CUDA")
        netD.cuda()
    return netD

import os
from torch.autograd import Variable

train_loader = load_data_CIFAR10()

# if not os.path.exists('./result'):
#     os.mkdir('result/')

# if not os.path.exists('./model'):
#     os.mkdir('model/')

def inception_eval(experiment, G):
    G.eval()

    images_gan = []

    z_dim = 100
    batch_size = 200
    with torch.no_grad():
        for i in range(100):
            z = Variable(torch.FloatTensor(batch_size, 100, 1, 1).normal_(0, 1).cuda())
            img = G(z).cpu()
            if i == 0:
                images = img
            else:
                images = torch.cat((images, img), dim = 0)
    images = images.view(-1,64,64,3)
    images = images.detach().cpu().numpy()
    print(images.shape)
    images=np.round((images+1)*(255/2))

    # important!
    torch.cuda.empty_cache()

    for x in images:
        images_gan.append(x)
    print("\nCalculating IS...")
    incept = get_inception_score(images_gan)
    torch.cuda.empty_cache()
    print(incept)
    experiment.log_metric("inception_score", incept[0])


def opt_experiment(experiment, netG):

    alpha = experiment.get_parameter("alpha")
    beta = experiment.get_parameter("beta")

    exp_parameters = {
        "data": "cifar10_64x64",
        "model": "64xDConv",
        "opt_gen": "Adam_lr_0.0002, (0,5,0.999)",
        "opt_dis": "Adam_lr_0.0002, (0.5,0.999)",
        "z_dim": 100,
        "n_critic": 1,
        "normalize": "mean,std 0.5",
        "dis_landscape": 0,
        "try": 0,
        "model_save": "NAofc"
    }

    experiment.log_parameters(exp_parameters)

    experiment.train()
        
    # netG = get_netG()
    netD1 = get_netD()
    netD2 = get_netD()

    # setup optimizer
    optimizerD1 = torch.optim.Adam(netD1.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizerD2 = torch.optim.Adam(netD2.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizerG = torch.optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))

    criterion_log = Log_loss()
    criterion_itself = Itself_loss()

    input = torch.FloatTensor(64, 3, 64, 64)
    noise = torch.FloatTensor(64, 100, 1, 1)
    fixed_noise = torch.FloatTensor(64, 100, 1, 1).normal_(0, 1)
    fixed_noise = Variable(fixed_noise)

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        criterion_log, criterion_itself = criterion_log.cuda(),  criterion_itself.cuda()
        input= input.cuda()
        noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

    output_temp = '.temp_d2.png'
    # path_to_save = "../models/cifar10_d2gan_64_lr2/dc64_ganns_"
    n_epoch = 150
    for epoch in range(0, n_epoch+1):
        netG.train()
        for i, data in enumerate(train_loader):
            real_cpu, _ = data
            batch_size = real_cpu.size(0)
            ######################################
            # train D1 and D2
            #####################################
            
            netD1.zero_grad()
            netD2.zero_grad()
            # train with real
            if use_cuda:
                real_cpu = real_cpu.cuda()
                
            input.resize_as_(real_cpu).copy_(real_cpu)        
            inputv = Variable(input)
            
            # D1 sees real as real, minimize -logD1(x)
            output = netD1(inputv)
            errD1_real = alpha * criterion_log(output)#criterion(output1, labelv) * 0.2
            errD1_real.backward()
            
            # D2 sees real as fake, minimize D2(x)
            output = netD2(inputv)
            errD2_real = criterion_itself(output, False)
            errD2_real.backward()
            
            # train with fake
            noise.resize_(batch_size, 100, 1, 1).normal_(0,1)
            noisev = Variable(noise)
            fake = netG(noisev)
            
            # D1 sees fake as fake, minimize D1(G(z))
            output = netD1(fake.detach())
            errD1_fake = criterion_itself(output, False)
            errD1_fake.backward()
            
            # D2 sees fake as real, minimize -log(D2(G(z))
            output = netD2(fake.detach())
            errD2_fake = beta * criterion_log(output)
            errD2_fake.backward()
            
            optimizerD1.step()
            optimizerD2.step()
            
            ##################################
            # train G
            ##################################
            netG.zero_grad()
            # G: minimize -D1(G(z)): to make D1 see fake as real
            output = netD1(fake)
            errG1 = criterion_itself(output)
            
            # G: minimize logD2(G(z)): to make D2 see fake as fake
            output = netD2(fake)
            errG2 = criterion_log(output, False)
            
            errG = errG2*beta + errG1
            errG.backward()
            optimizerG.step()
            

        print('[%d/%d]: loss_d1: %.3f, loss_d2: %.3f, loss_g: %.3f' % (
                (epoch), n_epoch, errD1_fake.item() + errD1_real.item(), errD2_fake.item() + errD2_real.item(), errG.item()))
        netG.eval()
        fake = netG(fixed_noise)
        if use_cuda:
            vutils.save_image(fake.detach().cpu(), output_temp, normalize=True)
        # if epoch%10 == 0:
        #     torch.save(netD1.state_dict(), path_to_save + str(epoch) + "_D1.pth")
        #     torch.save(netD2.state_dict(), path_to_save + str(epoch) + "_D2.pth")
        #     torch.save(netG.state_dict(), path_to_save + str(epoch) + "_G.pth")
        #     print("................checkpoint created...............")
        
        experiment.log_metric("D1_loss", (errD1_fake + errD1_real).cpu().item())
        experiment.log_metric("D2_loss", (errD2_fake + errD2_real).cpu().item())
        experiment.log_metric("G_loss", errG.item())
        if math.isnan(errG.item()) or math.isinf(errG.item()):
            return 0 
        # experiment.log_metric("critic_acc", torch.mean(torch.FloatTensor(D_accuracy)))

        experiment.log_image(output_temp, name = "output_" + str(epoch))
        if epoch%30 == 0:
            inception_eval(experiment, G)
    return 1

config = {
    "algorithm": "bayes",
    "name": "Optimize CIFAR10-D2GAN Network",
    "spec": {"maxCombo": 0, "objective": "maximize", "metric": "inception_score"},
    "parameters": {
        # "D_lr": {"type": "float", "min": 1e-4, "max": 2e-4},
        "alpha": {"type": "float", "min":0.1, "max": 0.99},
        "beta": {"type": "float", "min":0.1, "max": 0.99}
    },
    "trials": 1,
}

opt = comet_ml.Optimizer(config)

for experiment in opt.get_experiments(project_name="cifar_d2gan64_opt_incept4"):
    # Log parameters, or others:
    # experiment.log_parameter("epochs", 10)
    # init Generator
    G = get_netG()
    # Train it:
    c = opt_experiment(experiment, G)

    # How well did it do?
    if c:
        inception_eval(experiment, G)

    # Optionally, end the experiment:
    experiment.end()
    torch.cuda.empty_cache()

