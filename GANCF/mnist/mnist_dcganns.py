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

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device: ", device)

bs = 100

# MNIST Dataset
transform = transforms.Compose([
    transforms.Scale(64),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5), std=(0.5))])

train_dataset = datasets.MNIST(root='../datasets/mnist_data/', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='../datasets/mnist_data/', train=False, transform=transform, download=False)

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=bs, shuffle=False)

# dataset_ordered = []
# for i in range(10):
#   dataset_ordered.append(train_dataset.data[mnist.targets==i])

# dataloaders_ordered = []
# for dataset in dataset_ordered:
#   dataloaders_ordered.append(DataLoader(dataset, batch_size = args.batch_size, shuffle = True))

# DCGAN
class generator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    def __init__(self):
        #print('---------- generator -------------')
        super(generator, self).__init__()
        self.input_height = 64
        self.input_width = 64
        self.input_dim = 100
        self.output_dim = 1

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 1024 * (self.input_height // 16) * (self.input_width // 16)),
            # nn.BatchNorm1d(1024 * (self.input_height // 16) * (self.input_width // 16)),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 5, 2, 2, 1),
            # nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 5, 2, 2, 1),
            # nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 5, 2, 2, 1),
            # nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, self.output_dim, 5, 2, 2, 1),
            nn.Tanh(),
        )
        #utils.

    def forward(self, input):
        x = self.fc(input)
        x = x.view(-1, 1024, (self.input_height // 16), (self.input_width // 16))
        x = self.deconv(x)

        return x

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
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
    def __init__(self):
        super(discriminator, self).__init__()
        self.input_height = 64
        self.input_width = 64
        self.input_dim = 1
        self.output_dim = 1

        self.conv = nn.Sequential(
            nn.Conv2d(self.input_dim, 128, 5, 2, 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 5, 2, 2),
#            LayerNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 5, 2, 2),
#            LayerNorm2d(512),
            nn.LeakyReLU(0.2),     
            nn.Conv2d(512, 1024, 5, 2, 2),
#            LayerNorm2d(1024),
            nn.LeakyReLU(0.2),
        )
        self.fc = nn.Linear(1024 * (self.input_height // 16) * (self.input_width // 16), self.output_dim)
        
    def forward(self, input):
        x = self.conv(input)
        x = x.contiguous().view(-1, 1024 * (self.input_height // 16) * (self.input_width // 16))
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x

# build network
z_dim = 100
mnist_dim = train_dataset.data.size(1) * train_dataset.data.size(2)

G = generator().to(device)
D = discriminator().to(device)

# loss
criterion = nn.BCELoss() 

# optimizer
lr = 0.0001
G_optimizer = optim.Adam(G.parameters(), lr = lr, betas = (0.5, 0.999))
D_optimizer = optim.Adam(D.parameters(), lr = lr, betas = (0.5, 0.999))

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
    z = Variable(torch.randn(bs, z_dim).to(device))
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

    z = Variable(torch.randn(bs, z_dim).to(device))
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

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

experiment = comet_ml.Experiment(project_name="mnist_try_dcgan")

exp_parameters = {
    "data": "mnist_64x64",
    "model": "64xDConv, gen_tanh_noBN",
    "opt_gen": "Adam_lr_0.0001, (0,5,0.999)",
    "opt_dis": "Adam_lr_0.0001, (0.5,0.999)",
    "z_dim": 100,
    "n_critic": 1,
    "normalize": "mean:0.5, std:0.5",
    "dis_landscape": 0,
    "try": 3
}

experiment.log_parameters(exp_parameters)

experiment.train()

n_epoch = 200
n_critic = 1
for epoch in range(1, n_epoch+1):
    G.train()         
    D_losses, D_accuracy, G_losses = [], [], []
    for batch_idx, (x, _) in enumerate(train_loader):
        D_loss, D_acc = D_train(x)
        D_losses.append(D_loss)
        D_accuracy.append(D_acc)
        if batch_idx % n_critic == 0:
            G_losses.append(G_train(x))
        
    print('[%d/%d]: loss_d: %.3f, acc_d: %.3f, loss_g: %.3f' % (
            (epoch), n_epoch, torch.mean(torch.FloatTensor(D_losses)), torch.mean(torch.FloatTensor(D_accuracy)), torch.mean(torch.FloatTensor(G_losses))))

    # path_to_save = "../models/mnist_dcgan_2/dc64_ganns_"
    # if epoch%10 == 0:
    #     torch.save(D.state_dict(), path_to_save + str(epoch) + "_D.pth")
    #     torch.save(G.state_dict(), path_to_save + str(epoch) + "_G.pth")
    #     print("................checkpoint created...............")
    
    with torch.no_grad():
        G.eval()
        test_z = Variable(torch.randn(bs, z_dim).to(device))
        generated = G(test_z).cpu().data.numpy()[:64]

        experiment.log_metric("critic_loss", torch.mean(torch.FloatTensor(D_losses)))
        experiment.log_metric("gen_loss", torch.mean(torch.FloatTensor(G_losses)))
        experiment.log_metric("critic_acc", torch.mean(torch.FloatTensor(D_accuracy)))

        fig = plt.figure(figsize=(8, 8))
        gs = gridspec.GridSpec(8, 8)
        gs.update(wspace=0.05, hspace=0.05)

        for i, sample in enumerate(generated):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(sample.reshape(64,64), cmap='Greys_r')

        experiment.log_figure(figure=plt, figure_name = "figure_" + str(epoch))
        plt.close()     
        
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
