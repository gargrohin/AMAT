import comet_ml
comet_ml.config.save(api_key="CX4nLhknze90b8yiN2WMZs9Vw")

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
from inception import get_inception_score

from six.moves import urllib
# import tensorflow as tf
import glob
import scipy.misc
import math
import sys

from scipy.stats import entropy

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device: ", device)

bs = 64

# MNIST Dataset
transform = transforms.Compose([
    transforms.Scale(32),
    transforms.ToTensor(),
#    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))])

dataset = datasets.CIFAR10(root='../datasets/cifar10_data/', transform=transform, download=True)

# Data Loader (Input Pipeline)
dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=bs, shuffle=True, num_workers=2)

# dataset_ordered = []
# for i in range(10):
#   dataset_ordered.append(train_dataset.data[mnist.targets==i])

# dataloaders_ordered = []
# for dataset in dataset_ordered:
#   dataloaders_ordered.append(DataLoader(dataset, batch_size = args.batch_size, shuffle = True))

# DCGAN
#32x32
class generator(nn.Module):
    def __init__(self, ch = 512, bw=4):
        super(generator, self).__init__()
        self.ch = ch
        self.bw = bw
        self.lin = nn.Linear(z_dim, 4*4*ch)
        self.bn0 = nn.BatchNorm1d(4*4*ch)
        self.main = nn.Sequential(

            nn.ConvTranspose2d(ch, ch//2, 4, 2, 1),
            nn.BatchNorm2d(ch//2),
            nn.ReLU(True),

            # state size. (ngf*8) x 2x2
            nn.ConvTranspose2d(ch//2, ch//4, 4, 2, 1),
            nn.BatchNorm2d(ch//4),
            nn.ReLU(True),
            # state size. (ngf*4) x 4x4
            nn.ConvTranspose2d(ch//4, ch//8, 4, 2, 1),
            nn.BatchNorm2d(ch//8),
            nn.ReLU(True),
            # state size. (ngf*2) x 8x8
            nn.ConvTranspose2d(ch//8, 3, 3, 1, 1),
            # nn.BatchNorm2d(3),
            nn.Tanh(),
            # state size. (ngf) x 16x16
            # nn.ConvTranspose2d(ngf, nc, 4, 2, 1),
            # nn.Tanh()
            # state size. (nc) x 32x32
        )

    def forward(self, input):
        # if input.is_cuda and self.ngpu > 1:
        #     output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        # else:
        output = self.bn0(self.lin(input)).view(-1,self.ch, self.bw, self.bw)
        output = self.main(output)
        return output

class LayerNorm2d(nn.Module):
    """Layer for 2D layer normalization in CNNs.

    PyTorch's LayerNorm layer only works on the last channel, but PyTorch uses
    NCHW ordering in images. This layer moves the channel axis to the end,
    applies layer-norm, then permutes back.
    """

    def __init__(self, out_channels):
        """Initialize the child layer norm layer."""
        super().__init__()
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, inputs):
        """Apply layer normalization."""
        inputs = inputs.permute(0, 2, 3, 1)
        normed = self.norm(inputs)
        outputs = normed.permute(0, 3, 1, 2)
        return outputs

#DCGAN
class discriminator(nn.Module):
    def __init__(self, bw=4, ch=512):

        super(discriminator, self).__init__()
        self.bw = bw
        self.ch = ch
        self.main = nn.Sequential(
            # input is (nc) x 32x32
            nn.Conv2d(nc, ch//8, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 16x16
            nn.Conv2d(ch//8, ch//4, 4, 2, 1),
            nn.BatchNorm2d(ch//4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 8x8
            nn.Conv2d(ch//4, ch//4, 3, 1, 1),
            nn.BatchNorm2d(ch//4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 4x4
            nn.Conv2d(ch//4, ch//2, 4, 2, 1),
            nn.BatchNorm2d(ch//2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 2x2
            nn.Conv2d(ch//2, ch//2, 3, 1, 1),
            nn.BatchNorm2d(ch//2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ch//2, ch//1, 4, 2, 1),
            nn.BatchNorm2d(ch//1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ch//1, ch//1, 3, 1, 1),
            nn.BatchNorm2d(ch//1),
            nn.LeakyReLU(0.2, inplace=True),

        )
        self.ln = nn.Linear(bw*bw*ch, 1)
    
    def forward(self, input):
        # if input.is_cuda and self.ngpu > 1:
        #     output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        # else:
        output = self.main(input)
        return self.ln(output.view(-1,self.bw*self.bw*self.ch))


# build network
z_dim = 128
nc = 3
ngf = 64
ndf = 64
# mnist_dim = train_dataset.data.size(1) * train_dataset.data.size(2)

def weights_init_normal(m):
    # print(m)
    if type(m) == nn.Linear or type(m) == nn.ConvTranspose2d or type(m) == nn.Conv2d:
        m.weight.data.normal_(0.0, 0.05)
        # m.bias.data.fill_(0)

G = generator().to(device)
G.apply(weights_init_normal)
D = discriminator().to(device)
D.apply(weights_init_normal)

# loss
criterion = nn.BCELoss() 

# optimizer
lr = 0.0001
G_optimizer = optim.Adam(G.parameters(), lr = lr*2, betas = (0.0, 0.999))
D_optimizer = optim.Adam(D.parameters(), lr = lr*2, betas = (0.0, 0.999))

def D_train(x):
    #=======================Train the discriminator=======================#
    D.zero_grad()

    # train discriminator on real
    x_real, y_real = x, torch.ones(x.size()[0], 1)
    x_real, y_real = Variable(x_real.to(device)), Variable(y_real.to(device))

    D_output_real = D(x_real)
    D_real_loss = torch.mean(F.softplus(-D_output_real))
    D_real_score = D_output_real

    # train discriminator on facke
    z = Variable(torch.randn(bs, z_dim).to(device))
    x_fake, y_fake = G(z), Variable(torch.zeros(bs, 1).to(device))

    D_output_fake = D(x_fake)
    D_fake_loss = torch.mean(F.softplus(D_output_fake))
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

    z = Variable(torch.randn(bs, z_dim).to(device))
    y = Variable(torch.ones(bs, 1).to(device))

    G_output = G(z)
    D_output = D(G_output)
    G_loss = torch.mean(F.softplus(-D_output))

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

# def inception_eval(experiment, G):
    
#     G.eval()

#     images_gan = list()

#     z_dim = 128
#     batch_size = 64
#     with torch.no_grad():
#         for i in range(300):
#             z = Variable(torch.randn(batch_size, z_dim).to(device))
#             img = G(z).cpu()
#             gen_imgs = img.mul_(127.5).add_(127.5).clamp_(0.0, 255.0).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
#             images_gan.extend(list(gen_imgs))
#     print(len(images_gan))

#     # important!
#     torch.cuda.empty_cache()

#     print("\nCalculating IS...")
#     incept = get_inception_score(images_gan)
#     print(incept)
#     experiment.log_metric("inception_score", incept[0])

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

experiment = comet_ml.Experiment(project_name="cifar10_dcchainer")

exp_parameters = {
    "data": "cifar10_32x32",
    "model": "32x32_chainer",
    "opt_gen": "Adam_lr_0.0002, (0.0,0.999)",
    "opt_dis": "Adam_lr_0.0002, (0.0,0.999)",
    "z_dim": 128,
    "batch_size": 64,
    "n_critic": 1,
    "normalize": "mean,std 0.5",
    "dis_landscape": 0,
    "try": 1,
    "model_save": "cifar10_chainer_init"
}

experiment.log_parameters(exp_parameters)

experiment.train()

output = '.temp_ch1.png'
n_epoch = 200
n_critic = 1
for epoch in range(1, n_epoch+1):
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

    path_to_save = "../models/cifar10_chainer_init/dc64_ganns_"
    if epoch%10 == 0:
        torch.save(D.state_dict(), path_to_save + str(epoch) + "_D.pth")
        torch.save(G.state_dict(), path_to_save + str(epoch) + "_G.pth")
        print("................checkpoint created...............")
    
    with torch.no_grad():
        G.eval()
        test_z = Variable(torch.randn(64, z_dim).to(device))
        generated = G(test_z).detach().cpu()

        experiment.log_metric("critic_loss", torch.mean(torch.FloatTensor(D_losses)))
        experiment.log_metric("gen_loss", torch.mean(torch.FloatTensor(G_losses)))
        experiment.log_metric("critic_acc", torch.mean(torch.FloatTensor(D_accuracy)))

        vutils.save_image(generated, output ,normalize=True)

        experiment.log_image(output, name = "output_" + str(epoch))

#        if epoch == 1 or epoch%20 == 0 or epoch==n_epoch:
#            inception_eval(experiment, G)

        
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
