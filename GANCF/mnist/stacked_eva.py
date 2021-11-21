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
from torch.utils.data import Dataset, DataLoader

import os.path
import sys
import tarfile
from scipy.stats import entropy

import keras

from PIL import Image

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device: ", device)

bs = 100

# MNIST Dataset
transform = transforms.Compose([
    transforms.Scale(64),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5), std=(0.5))])

def evaluate(x, model):
    output = model.predict(x)
    return list(np.argmax(output, axis=1))

dataset = datasets.MNIST(root='../datasets/mnist_data/', transform=transform, download=True)

class stackedMNIST(Dataset):

    def __init__(self, dataset, transform=None):
        
        self.data = dataset.data
        self.targets = dataset.targets
        self.transform = transform
        self.total = dataset.targets.size()[0]
        self.rgbsize = self.total

    def __len__(self):
        return self.rgbsize

    def __getitem__(self, idx):

        np.random.seed(idx)

        rgb = []
        for i in range(3):
            ind = np.random.randint(self.total)
            rgb.append(self.data[ind])
            rgb[i] = Image.fromarray(rgb[i].numpy(), mode='L')
        rgb_trans = []
        for i in range(3):
            rgb_trans.append(transform(rgb[i]))
        flag = True
        for i in range(3):
            if flag:
                img = rgb_trans[i]
                flag = False
            else:
                img = torch.cat((img, rgb_trans[i]), dim = 0)
        
        return img

dataset = stackedMNIST(dataset, transform)

# Data Loader (Input Pipeline)
dataloader = DataLoader(dataset=dataset, batch_size=bs, shuffle=True, num_workers=2)

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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device: ", device)

nc = 3
ngf = 64
z_dim = 100
ndf = 64
G = generator().to(device)

gen_ind = 1

#path_G = "../../GANCF/models/cifar10_dcgan_1/dc64_ganns_" + str(gen_ind) + "_G.pth"
path_G = "../models/stacked_multi_0/dc64_ganns_" + str(gen_ind) + "_G.pth"
G.load_state_dict(torch.load(path_G))
G.eval()

modes = np.zeros(1000)

transform_eva = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Scale(28),
    transforms.ToTensor()])

batch_size = 200
with torch.no_grad():
    for i in range(100):
        z = Variable(torch.randn(batch_size, z_dim, 1, 1).to(device))
        img = G(z).cpu()
        if i == 0:
            images = img
        else:
            images = torch.cat((images, img), dim = 0)

print()
print(images.size())
print()

torch.cuda.empty_cache()

model = keras.models.load_model("../../mnist_model.hdf5")

for b in images:
    b = (b+1)/2
    b = transform_eva(b)
    r = evaluate(b.view(3,28,28,1), model)
    r = 100*r[0] + 10*r[1] + r[2]
    modes[r.item()]+=1

total_modes = 0
for m in modes:
    if m:
        total_modes+=1

print()
print(total_modes)

modes_d = np.zeros(1000)
for x in dataloader:
    for b in x:
        b = (b+1)/2
        b = transform_eva(b)
        r = evaluate(b.view(3,28,28,1), model)
        r = 100*r[0] + 10*r[1] + r[2]
        modes_d[r.item()]+=1

print(entropy(modes/sum(modes), modes_d/sum(modes_d)))
print(entropy(modes_d/sum(modes_d), modes/sum(modes)))
