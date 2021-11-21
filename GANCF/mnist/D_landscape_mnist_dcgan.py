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
import matplotlib.pyplot as plt
import os

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device: ", device)

bs = 1

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

# DCGAN
class generator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    def __init__(self):
        #print('---------- generator -------------')
        super(generator, self).__init__()
        self.input_height = 64
        self.input_width = 64
        self.input_dim = 25
        self.output_dim = 1

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 1024 * (self.input_height // 16) * (self.input_width // 16)),
            nn.BatchNorm1d(1024 * (self.input_height // 16) * (self.input_width // 16)),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 5, 2, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 5, 2, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 5, 2, 2, 1),
            nn.BatchNorm2d(128),
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
        x = F.dropout(x, 0.3)
        x = self.fc(x)
        x = F.dropout(x, 0.3)
        x = torch.sigmoid(x)
        return x

# build network
z_dim = 25
mnist_dim = train_dataset.data.size(1) * train_dataset.data.size(2)

G = generator().to(device)
D1 = discriminator().to(device)
D2 = discriminator().to(device)

# loss
criterion = nn.BCELoss() 

path_G = "../models/mnist_dcgan/dc64_ganns_200_G.pth"
path_D = "../models/mnist_dcgan/dc64_ganns_200_D.pth"
G.load_state_dict(torch.load(path_G))
G.eval()

D1.load_state_dict(torch.load(path_D))
D1.eval()
path_D = "../models/mnist_dcgan/dc64_ganns_10_D.pth"
D2.load_state_dict(torch.load(path_D))
D2.eval()

experiment = comet_ml.Experiment(project_name="mnist_try_dcgan_landscape")

exp_parameters = {
    "data": "mnist_64x64",
    "model": "64xDConv, gen_tanh, dis_dropout",
    "opt_gen": "Adam_lr_0.0002, (0,5,0.999)",
    "opt_dis": "Adam_lr_0.0002, (0.5,0.999)",
    "z_dim": 100,
    "normalize": "mean:0.5, std:0.5",
    "dis_landscape": 1
}

experiment.log_parameters(exp_parameters)


# Sample data to get discriminator landscape.
samples = []
done = []
for i in range(10):
    done.append(0)
    samples.append([])
for x,y in train_loader:
    if done[y[0]] < 64:
        samples[y[0]].append(x[0])
        done[y[0]]+=1
for i in range(10):
    samples[i] = torch.cat(samples[i], dim = 0).unsqueeze(1)

D_outs = []
for img in samples:
    # img = torch.FloatTensor(img)
    print(img.size())
    D_outs.append(torch.mean(D1(img.float().cuda())).cpu().detach().numpy())

plt.ylim((0,1))
for i in range(10):
    plt.scatter(i,D_outs[i], c = 'g')
    print(i , D_outs[i])
plt.title("D_modes")
experiment.log_figure(figure=plt, figure_name = "D1_modes_realdata")
plt.close()


D_outs = []
for img in samples:
    img = torch.FloatTensor(img)
    D_outs.append(torch.mean(D2(img.float().cuda())).cpu().detach().numpy())

plt.ylim((0,1))
for i in range(10):
    plt.scatter(i,D_outs[i], c = 'g')
    print(i , D_outs[i])
plt.title("D_modes")
experiment.log_figure(figure=plt, figure_name = "D2_modes_realdata")
plt.close()

