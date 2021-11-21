import comet_ml
import logging
comet_ml.config.save(api_key="CX4nLhknze90b8yiN2WMZs9Vw")
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("comet_ml")

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"

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

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device: ", device)

bs = 128

# MNIST Dataset
transform = transforms.Compose([
    transforms.Scale(32),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2466, 0.2431, 0.2610])])
    transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))])

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
#32x32
class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(z_dim, ngf * 8, 2, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 2x2
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 4x4
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 8x8
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 16x16
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # nn.Sigmoid()
            # state size. (nc) x 32x32
        )

    def forward(self, input):
        # if input.is_cuda and self.ngpu > 1:
        #     output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        # else:
        output = self.main(input)
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
    def __init__(self):
        super(discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 32x32
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 16x16
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 8x8
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 4x4
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 2x2
            nn.Conv2d(ndf * 8, 1, 2, 1, 0, bias=False),
            # nn.Sigmoid()
        )

    def forward(self, input):
        # if input.is_cuda and self.ngpu > 1:
        #     output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        # else:
        output = self.main(input)

        return output.view(-1, 1).squeeze(1)
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
    images = images.view(-1,32,32,3)
    images = images.detach().cpu().numpy()
    print(images.shape)
    images=np.round((images+1)*(255/2))
    # images=np.round((images)*(255))
    
    # for ch in range(3):
    #     mini = np.min(images[:,:,:,ch])
    #     images[:,:,:,ch] = images[:,:,:,ch] - mini
    #     maxi = np.max(images[:,:,:,ch])
    #     images[:,:,:,ch]/=maxi
    
    images = np.round(images*255)

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

    D = discriminator().to(device)

    # loss
    criterion = nn.BCELoss() 

    # optimizer
    lr = 0.0001
    # D_lr = experiment.get_parameter("D_lr")

    G_optimizer = optim.Adam(G.parameters(), lr = lr*2, betas = (0.0, 0.999))
    D_optimizer = optim.Adam(D.parameters(), lr = lr*2, betas = (0.0, 0.999))

    def D_train(x):
        #=======================Train the discriminator=======================#
        D.zero_grad()

        # train discriminator on real
        x_real, y_real = x, torch.ones(x.size()[0], 1)
        x_real, y_real = Variable(x_real.to(device)), Variable(y_real.to(device))

        D_output_real = D(x_real)
        # D_real_loss = criterion(D_output_real, y_real)
        # D_real_score = D_output_real

        # train discriminator on facke
        z = Variable(torch.randn(bs, z_dim, 1, 1).to(device))
        x_fake, y_fake = G(z), Variable(torch.zeros(bs, 1).to(device))

        D_output_fake = D(x_fake)
        # D_fake_loss = criterion(D_output_fake, y_fake)
        # D_fake_score = D_output_fake

        D_acc = get_critic_acc(D_output_fake, D_output_real)

        # gradient backprop & optimize ONLY D's parameters
        # D_loss = D_real_loss + D_fake_loss
        D_loss = torch.mean(nn.ReLU(inplace=True)(1.0 - D_output_real)) + \
                 torch.mean(nn.ReLU(inplace=True)(1 + D_output_fake))
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
        # G_loss = criterion(D_output, y)
        G_loss = -torch.mean(D_output)

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
        "data": "cifar10_32x32",
        "loss": "hinge",
        "model": "32xDConv dis_noactivation",
        "opt_gen": "Adam_lr_0.0002, (0,0,0.999)",
        "opt_dis": "Adam_lr_0.0002, (0.0,0.999)",
        # "dis_lr": D_lr,
        "z_dim": 100,
        # "n_critic": 1,
        "normalize": "mean,std 0.5",
        "dis_landscape": 0,
        "try": 0,
        "model_save": "NAofc"
    }

    experiment.log_parameters(exp_parameters)
    experiment.train()

    output = '.temp_32_h.png'
    n_epoch = 150
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
            if epoch%30 == 0:
                inception_eval(experiment, G)


config = {
    "algorithm": "bayes",
    "name": "Optimize CIFAR10-DCGAN Network",
    "spec": {"maxCombo": 0, "objective": "maximize", "metric": "inception_score"},
    "parameters": {
        # "D_lr": {"type": "float", "min": 5e-5, "max": 2e-4},
        "n_critic": {"type": "integer", "min": 0, "max": 5},
    },
    "trials": 1,
}

opt = comet_ml.Optimizer(config)

for experiment in opt.get_experiments(project_name="cifar_dchinge_opt_incept1"):
    # Log parameters, or others:
    # experiment.log_parameter("epochs", 10)
    # init Generator
    G = generator().to(device)
    # Train it:
    opt_experiment(experiment, G)

    # How well did it do?
    inception_eval(experiment, G)

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
