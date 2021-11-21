import comet_ml
import logging
comet_ml.config.save(api_key="CX4nLhknze90b8yiN2WMZs9Vw")
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("comet_ml")

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

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
import argparse


from six.moves import urllib
import tensorflow as tf
import glob
import scipy.misc
import math
import sys

from scipy.stats import entropy

from inception import get_inception_score

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device: ", device)

bs = 128

# MNIST Dataset
transform = transforms.Compose([
    transforms.Scale(64),
    transforms.ToTensor()])
    # transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2466, 0.2431, 0.2610])])
    # transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))])

dataset = datasets.CIFAR10(root='../datasets/cifar10_data/', transform=transform, download=True)

# Data Loader (Input Pipeline)
dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=bs, shuffle=True)

# dataset_ordered = []
# for i in range(10):
#   dataset_ordered.append(train_dataset.data[mnist.targets==i])

# dataloaders_ordered = []
# for dataset in dataset_ordered:
#   dataloaders_ordered.append(DataLoader(dataset, batch_size = args.batch_size, shuffle = True))

# DCGAN
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


def parse_option():

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--eval_freq', type=int, default=1, help='meta-eval frequency')
    parser.add_argument('--save_freq', type=int, default=10, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=200, help='number of training epochs')
    parser.add_argument('--z_dim', type=int, default=100, help='embedding dimension for transformer')
    parser.add_argument('--n_dis', type=int, default=1, help='number of discrimnators')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.0002, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='15', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--LUT_lr', default=[(20, 0.05), (30, 0.01), (40, 0.006), (50, 0.001), (60, 0.0001)], help="multistep to decay learning rate")
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--tags', type=str, default="gen0, ssl", help='add tags for the experiment')


    # specify folder
    parser.add_argument('--model_path', type=str, default='save/DCGAN', help='path to save model')

    opt = parser.parse_args()

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))
        
    tags = opt.tags.split(',')
    opt.tags = list([])
    for it in tags:
        opt.tags.append(it)

    opt.model_name = 'DCGAN'

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt
opt = parse_option()

z_dim = 100
nc = 3
ngf = 64
ndf = 64

def inception_eval(experiment, G):
    
    G.eval()

    images_gan = []

    z_dim = 100
    batch_size = 200
    with torch.no_grad():
        for i in range(100):
            z = Variable(torch.randn(batch_size, z_dim, 1, 1).to(device))
            img = G(z).cpu()
            if i == 0:
                images = img
            else:
                images = torch.cat((images, img), dim = 0)
    images = images.view(-1,64,64,3)
    images = images.detach().cpu().numpy()
    print(images.shape)
    images=np.round((images+1)*(255/2))
    images=np.round((images)*(255))
    
    # for ch in range(3):
    #     mini = np.min(images[:,:,:,ch])
    #     images[:,:,:,ch] = images[:,:,:,ch] - mini
    #     maxi = np.max(images[:,:,:,ch])
    #     images[:,:,:,ch]/=maxi
    
    # images = np.round(images*255)

    # important!
    torch.cuda.empty_cache()

    for x in images:
        images_gan.append(x)

    print("\nCalculating IS...")
    incept = get_inception_score(images_gan)
    print(incept)
    experiment.log_metric("inception_score", incept[0])


def opt_experiment(experiment, G):
    # build network

    D = Discriminator(opt).to(device)

    # loss
    criterion = nn.BCELoss() 

    # optimizer
    lr = 0.0001
    D_lr = lr

    G_optimizer = optim.Adam(G.parameters(), lr = lr*2, betas = (0.0, 0.999))
    D_optimizer = optim.Adam(D.parameters(), lr = lr*2, betas = (0.0, 0.999))

    def D_train(x):
        #=======================Train the discriminator=======================#
        D.zero_grad()

        # train discriminator on real
        x_real, y_real = x, torch.ones(x.size()[0], 1)
        x_real, y_real = Variable(x_real.to(device)), Variable(y_real.to(device))

        D_output_real = D(x_real)
        D_real_loss = criterion(D_output_real, y_real)
        D_real_score = D_output_real

        # train discriminator on facke
        z = Variable(torch.randn(bs, z_dim, 1, 1).to(device))
        x_fake, y_fake = G(z), Variable(torch.zeros(bs, 1).to(device))

        D_output_fake = D(x_fake)
        D_fake_loss = criterion(D_output_fake, y_fake)
        D_fake_score = D_output_fake

        D_acc = get_critic_acc(D_output_fake, D_output_real)

        # gradient backprop & optimize ONLY D's parameters
        D_loss = D_real_loss + D_fake_loss
        D_loss.backward()
        D_optimizer.step()
            
        return  D_loss.data.item(), D_acc

    def G_train(x):
        #=======================Train the generator=======================#
        G.zero_grad()

        z = Variable(torch.randn(bs, z_dim, 1, 1).to(device))
        y = Variable(torch.ones(bs, 1).to(device))

        G_output = G(z)
        D_output = D(G_output)
        G_loss = criterion(D_output, y)

        # gradient backprop & optimize ONLY G's parameters
        G_loss.backward()
        G_optimizer.step()
            
        return G_loss.data.item()

    def get_critic_acc(critic_fake, critic_real):
        acc = 0.0
        for x in critic_fake[:]:
            if x.item() <= 0.5:
                acc = acc + 1
        for x in critic_real[:]:
            if x.item() >= 0.5:
                acc = acc + 1
        
        acc = acc/(critic_fake.size()[0] + critic_real.size()[0])
        return acc

    # import matplotlib.pyplot as plt
    # import matplotlib.gridspec as gridspec

    exp_parameters = {
        "data": "cifar10_64",
        "model": "init everything",
        "opt_gen": "Adam_lr_0.0002, (0,5,0.999)",
        "opt_dis": "Adam_lr_0.0002 (0.5,0.999)",
        # "dis_lr": 'init',
        "z_dim": 100,
        "n_critic": 1,
        "normalize": "mean,std none",
        "dis_landscape": 0,
        "try": 0,
        "model_save": "NAofc"
    }

    experiment.log_parameters(exp_parameters)
    experiment.train()

    output = '.temp_64.png'
    n_epoch = 200
    n_critic = 1
    for epoch in range(0, n_epoch+1):
        G.train()
        D_losses, D_accuracy, G_losses = [], [], []
        for batch_idx, (x, _) in enumerate(dataloader):
            D_loss, D_acc = D_train(x)
            D_losses.append(D_loss)
            D_accuracy.append(D_acc)
            if batch_idx % n_critic == 0:
                G_losses.append(G_train(x))
            
        print('[%d/%d]: loss_d: %.3f, acc_d: %.3f, loss_g: %.3f' % (
                (epoch), n_epoch, torch.mean(torch.FloatTensor(D_losses)), torch.mean(torch.FloatTensor(D_accuracy)), torch.mean(torch.FloatTensor(G_losses))))

        # path_to_save = "../models/cifar10_dcgan_32_lr2/dc64_ganns_"
        # if epoch%10 == 0:
        #     torch.save(D.state_dict(), path_to_save + str(epoch) + "_D.pth")
        #     torch.save(G.state_dict(), path_to_save + str(epoch) + "_G.pth")
        #     print("................checkpoint created...............")
        
        with torch.no_grad():
            G.eval()
            test_z = Variable(torch.randn(64, z_dim, 1, 1).to(device))
            generated = G(test_z).detach().cpu()

            experiment.log_metric("critic_loss", torch.mean(torch.FloatTensor(D_losses)))
            experiment.log_metric("gen_loss", torch.mean(torch.FloatTensor(G_losses)))
            experiment.log_metric("critic_acc", torch.mean(torch.FloatTensor(D_accuracy)))

            vutils.save_image(generated, output ,normalize=True)

            experiment.log_image(output, name = "output_" + str(epoch))
            if epoch%5 == 0:
                inception_eval(experiment, G)


# config = {
#     "algorithm": "bayes",
#     "name": "Optimize CIFAR10-DCGAN Network",
#     "spec": {"maxCombo": 0, "objective": "maximize", "metric": "inception_score"},
#     "parameters": {
#         "D_lr": {"type": "float", "min": 6e-5, "max": 2e-4},
#     },
#     "trials": 1,
# }

# opt = comet_ml.Optimizer(config)
experiment = comet_ml.Experiment(project_name="cifar10_dcgan_j1")

# for experiment in opt.get_experiments(project_name="cifar_dcgan_opt_incept2"):
# Log parameters, or others:
# experiment.log_parameter("epochs", 10)
# init Generator
G = Generator(opt).to(device)
# Train it:
opt_experiment(experiment, G)

# How well did it do?
# inception_eval(experiment, G)

# Optionally, end the experiment:
experiment.end()


# # Sample data to get discriminator landscape.
# samples = []
# done = []
# for i in range(10):
#     done.append(0)
# for x,y in train_loader:
#     if done[y[0]] == 0:
#         samples.append(x[0])
#         done[y[0]]=1

# print()
# print(samples[0].size())

# D_outs = []
# # for dataloader in dataloaders_ordered:
# for img in samples:
#     D_outs.append(torch.mean(D(img.view(-1, mnist_dim).float().cuda())).cpu().detach().numpy())

# plt.ylim((0,1))
# for i in range(10):
#     plt.scatter(i,D_outs[i], c = 'g')
#     print(i , D_outs[i])
# experiment.log_figure(figure=plt, figure_name = "dis_modes")
# plt.close()
