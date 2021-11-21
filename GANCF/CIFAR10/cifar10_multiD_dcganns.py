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

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device: ", device)

bs = 128

# MNIST Dataset
transform = transforms.Compose([
    transforms.Scale(64),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
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
class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(z_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
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
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        # if input.is_cuda and self.ngpu > 1:
        #     output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        # else:
        output = self.main(input)

        return output.view(-1, 1).squeeze(1)


# build network
z_dim = 100
nc = 3
ngf = 64
ndf = 64
n_dis = 2
# mnist_dim = train_dataset.data.size(1) * train_dataset.data.size(2)

G = generator().to(device)
# optimizer
lr = 0.0001
G_optimizer = optim.Adam(G.parameters(), lr = lr*5, betas = (0.5, 0.999))
#multi D
multiD = []
multiD_optim = []
for d in range(n_dis):
    multiD.append(discriminator().to(device))
    multiD_optim.append(optim.Adam(multiD[d].parameters(), lr = lr/2, betas = (0.5, 0.999)))
# loss
criterion = nn.BCELoss() 

def D_train(x_real):
    #=======================Train the discriminator=======================#
    G.train()
    for i in range(n_dis):
        multiD[i].train()
        for p in multiD[i].parameters():
            p.requires_grad = True
        multiD_optim[i].zero_grad()

    for p in G.parameters():
        p.requires_grad = True
    
    flag = True
    z = Variable(torch.randn(x_real.size()[0], z_dim, 1, 1).to(device))
    x_fake = G(z)
    x_real = Variable(x_real.to(device))
    for i in range(n_dis):
        if flag:
            D_fake = multiD[i](x_fake).unsqueeze(1)
            D_real = multiD[i](x_real).unsqueeze(1)
            flag = False
        else:
            D_fake = torch.cat((D_fake, multiD[i](x_fake).unsqueeze(1)), dim = 1)
            D_real = torch.cat((D_real, multiD[i](x_real).unsqueeze(1)), dim = 1)
    
    ind = torch.argmin(D_fake, dim = 1)
    mask = torch.zeros((x_real.size()[0], n_dis)).to(device)

    for i in range(mask.size()[0]):
        random_checker = np.random.randint(0,10)
        if random_checker > 7:
            index = np.random.randint(0,n_dis)
            mask[i][index] = 1.0
        else:
            mask[i][ind[i]] = 1.0
    
    D_fake_output = torch.sum(mask*D_fake, dim = 1)
    D_real_output = torch.sum(mask*D_real, dim = 1)

    y_real = Variable(torch.ones(x_real.size()[0], 1).to(device), requires_grad = False)
    y_fake = Variable(torch.zeros(x_real.size()[0], 1).to(device), requires_grad = False)
    
    D_real_loss = criterion(D_real_output, y_real)
    D_fake_loss = criterion(D_fake_output, y_fake)

    D_acc = get_critic_acc(D_fake_output, D_real_output)

    # gradient backprop & optimize ONLY D's parameters
    D_loss = D_real_loss + D_fake_loss
    D_loss.backward()

    for i in range(n_dis):
        if i in ind:
            multiD_optim[i].step()
    # multiD_optim[dis_index].step()
        
    return  D_loss.data.item(), D_acc, ind

def G_train(x):
    #=======================Train the generator=======================#
    G.train()
    G_optimizer.zero_grad()

    z = Variable(torch.randn(bs, z_dim, 1, 1).to(device))
    y = Variable(torch.ones(bs, 1).to(device), requires_grad = False)

    critic_fakes = []
    fake_img = G(z)
    lit = np.zeros(n_dis)
    for i in range(n_dis):
        for p in multiD[i].parameters():
            p.requires_grad = False
        critic_fake = multiD[i](fake_img)
        critic_fakes.append(critic_fake)
        lit[i] = torch.sum(critic_fake).item()
    loss_sort = np.argsort(lit)
    weights = np.random.dirichlet(np.ones(n_dis))
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
    G_loss = criterion(critic_fake, y)

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

experiment = comet_ml.Experiment(project_name="cifar10_multiD_dcgan_1")

exp_parameters = {
    "data": "cifar10_64x64",
    "model": "64xDConv, gen_tanh",
    "opt_gen": "Adam_lr_0.0005, (0.5,0.999)",
    "opt_dis": "Adam_lr_0.00005, (0.5,0.999)",
    "n_dis": 2, 
    "z_dim": 100,
    "n_critic": 1,
    "normalize": "mean:0.5, std:0.5",
    "model_save": "cifar10_multid_2",
    "dis_landscape": 0,
    "try": 0,
    # "D update": "ndis 1 try whatever argmin over each datapoint using mask",
    "D update": "argmin over each fake datapoint using mask",
}

experiment.log_parameters(exp_parameters)

experiment.train()

output = '.temp_0.png'
n_epoch = 200
n_critic = 1
for epoch in range(1, n_epoch+1):
    G.train()
    D_losses, D_accuracy, G_losses = [], [], []
    for batch_idx, (x, _) in enumerate(dataloader):
        D_loss, D_acc, dis_index = D_train(x)
        D_losses.append(D_loss)
        D_accuracy.append(D_acc)
        if batch_idx % n_critic == 0:
            G_losses.append(G_train(x))
        
    print('[%d/%d]: loss_d: %.3f, acc_d: %.3f, loss_g: %.3f' % (
            (epoch), n_epoch, torch.mean(torch.FloatTensor(D_losses)), torch.mean(torch.FloatTensor(D_accuracy)), torch.mean(torch.FloatTensor(G_losses))))
    print("\ndis_index current: ", dis_index)

    path_to_save = "../models/cifar10_multid_2/dc64_ganns_"
    if epoch%10 == 0:
    #    torch.save(D.state_dict(), path_to_save + str(epoch) + "_D.pth")
       torch.save(G.state_dict(), path_to_save + str(epoch) + "_G.pth")
       print("................checkpoint created...............")
    
    with torch.no_grad():
        G.eval()
        test_z = Variable(torch.randn(64, z_dim, 1, 1).to(device))
        generated = G(test_z).detach().cpu()

        experiment.log_metric("critic_loss", torch.mean(torch.FloatTensor(D_losses)))
        experiment.log_metric("gen_loss", torch.mean(torch.FloatTensor(G_losses)))
        experiment.log_metric("critic_acc", torch.mean(torch.FloatTensor(D_accuracy)))

        vutils.save_image(generated, output ,normalize=True)

        experiment.log_image(output, name = "output_" + str(epoch))
        
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
