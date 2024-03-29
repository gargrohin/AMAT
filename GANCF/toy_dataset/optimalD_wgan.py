# -*- coding: utf-8 -*-
"""toy_dataset_gan_nonlinear.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1BqE7BnsCm3o-TPVjzEc692rKOnI1hhed
"""

import comet_ml
comet_ml.config.save(api_key="CX4nLhknze90b8yiN2WMZs9Vw")

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from torch.autograd import Variable
from copy import deepcopy

batch_size = 1000

import random

def gaussians(flag=True):  
  scale = 2.
  centers1 = [
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),
    (1. / np.sqrt(2), 1. / np.sqrt(2)),
    (1. / np.sqrt(2), -1. / np.sqrt(2)),
    (-1. / np.sqrt(2), 1. / np.sqrt(2)),
    (-1. / np.sqrt(2), -1. / np.sqrt(2))
  ]

  centers2 = []
  theta = np.pi/8
  a = np.cos(theta)
  b = np.sin(theta)
  for p in centers1:
    centers2.append((p[0]*a - p[1]*b, p[1]*a + p[0]*b))

  centers1 = [(scale * x, scale * y) for x, y in centers1]
  centers2 = [(scale * x, scale * y) for x, y in centers2]
  count = 0
  while True:
    count+=1
    dataset = []
    for i in range(batch_size):
      point = np.random.randn(2) * 0.05
      if flag:
        if count%50 == 0:
          center = random.choice(centers1)
        else:
          center = random.choice(centers2)
      else:
        center = random.choice(centers2)
      point[0] += center[0]
      point[1] += center[1]
      dataset.append(point)
    dataset = np.array(dataset, dtype='float32')
    dataset /= 1.414  # stdev
    yield dataset

data = gaussians(flag = False)
points = next(data)

all_points = np.copy(points)
dataset = np.copy(points)

cuda = True if torch.cuda.is_available() else False

import torch.nn.functional as F

class Wgen(nn.Module):
  
  def __init__(self, args):
    super(Wgen, self).__init__()

    self.model = nn.Sequential(
                 nn.Linear(args.Z_dim, args.dim1),
                #  nn.BatchNorm1d(args.dim1, 0.8),
                 nn.LeakyReLU(),
                 nn.Linear(args.dim1, args.dim2),
                #  nn.BatchNorm1d(args.dim2, 0.8),
                 nn.LeakyReLU(),
                 nn.Linear(args.dim2, args.dim3),
                #  nn.BatchNorm1d(args.dim3, 0.8),
                 nn.LeakyReLU(),
                 nn.Linear(args.dim3, args.out_dim),
                #  nn.Sigmoid,
                 nn.Tanh()
    )

  def forward(self, z):

    img_gen = self.model(z)
    return 2*img_gen

class Wdis(nn.Module):
  
  def __init__(self, args):
    super(Wdis, self).__init__()

    self.model = nn.Sequential(
                 nn.Linear(args.img_dim, args.dim1),
                #  nn.BatchNorm1d(args.dim1, 0.8),
                 nn.ReLU(),
                 nn.Linear(args.dim1, args.dim2),
                #  nn.BatchNorm1d(args.dim2, 0.8),
                 nn.ReLU(),
                 nn.Linear(args.dim2, 1),
                 nn.Sigmoid()
    )

  def forward(self, x):
    out = self.model(x)
    return out

import argparse

parser = argparse.ArgumentParser(description='CNN text classificer')
# learning
parser.add_argument('-lr', type=float, default=0.0001, help='initial learning rate [default: 0.001]')
parser.add_argument('-epochs', type=int, default=300, help='number of epochs for train [default: 256]')
parser.add_argument('-batch_size', type=int, default=128, help='batch size for training [default: 64]')
parser.add_argument('-Z_dim', type=int, default = 10)
parser.add_argument('-dim1', type=int, default = 128)
parser.add_argument('-dim2', type=int, default = 512)
parser.add_argument('-dim3', type=int, default = 1024)
parser.add_argument('-out_dim', type=int, default = 2)
parser.add_argument('-img_dim', type=int, default = 2)
# data 
parser.add_argument('-shuffle', action='store_true', default=True, help='shuffle the data every epoch')
# model
parser.add_argument('-dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
# device
parser.add_argument('-device', type=int, default=-1, help='device to use for iterate data, -1 mean cpu [default: -1]')
parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the gpu')

args = parser.parse_args(args=[])

gen = Wgen(args)
dis = Wdis(args)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

if cuda:
  print("Using cuda")
  gen.cuda()
  dis.cuda()

# gen_optim = optim.RMSprop(gen.parameters(), lr = args.lr)
# dis_optim = optim.RMSprop(dis.parameters(), lr = args.lr)

gen_optim = optim.Adam(gen.parameters(), lr = args.lr, betas = (0.5,0.999))
dis_optim = optim.Adam(dis.parameters(), lr = args.lr, betas = (0.5,0.999))

# gen_optim = optim.SGD(gen.parameters(), lr = args.lr*100, momentum = 0.0)
# dis_optim = optim.SGD(dis.parameters(), lr = args.lr*100, momentum = 0.0)

class gnn(nn.Module):
  
  def __init__(self):
    super(gnn, self).__init__()

    self.model = nn.Sequential(
                 nn.Linear(1, 10),
                 nn.LeakyReLU(),
                 nn.Linear(10, 100),
                 nn.LeakyReLU(),
                 nn.Linear(100,50),
                 nn.LeakyReLU(),
                 nn.Linear(50, 1),
                #  nn.Sigmoid,
                 nn.Tanh()
    )

  def forward(self, z):

    img_gen = self.model(z)
    return img_gen


functions = []

# gnns = []
# for i in range(10):
#   gnns.append(gnn())

path = '../../nonlinear_functions2.p'
f = open(path,'rb')
gnns = torch.load(f)
f.close()

with torch.no_grad():
  for i in range(int(len(gnns)/2)):
    # g1 = gnns[i*2]
    # g2 = gnns[i*2 +1]
    functions.append(lambda x: gnns[i*2](torch.FloatTensor(x).view(x.shape[0],1)).view(x.shape[0]).detach().numpy())
    functions.append(lambda x: gnns[i*2+1](torch.FloatTensor(x).view(x.shape[0],1)).view(x.shape[0]).detach().numpy())

# functions.append(lambda x: x)


def add_nonlinearity(x):

  i = 0
  for f in functions:
 
    i+=1
    i = i%2
    x[:,i] = x[:,i]*f(x[:,i-1])
      
  return x

def remove_nonlinearity(x):

  j = len(functions)
  eps = 10**(-10)
  i = j%2
  while j:
    j-=1
    f = functions[j]
    den = f(x[:,i-1])
    den[abs(den) < eps] = eps
    x[:,i] = x[:,i]/den
    i-=1
    i = i%2


  return x

# Transform matrix

A = (np.random.rand(1000, 2+1) -0.5 )*2  # keep A values in [-1,1]

def transform(x):
  # x = add_nonlinearity(x)
  x = np.vstack((x.T, np.ones(x.shape[0])))
  B = np.dot(A, x)
  B = B + np.random.normal(0 , 0 , B.shape)
  # B = add_nonlinearity(B)
  # B = B*B*B  
  return B.T

def detransform(x):
  # x = np.cbrt(x)
  # x = remove_nonlinearity(x)
  x = np.dot(np.dot(np.linalg.inv(np.dot(A.T,A)), A.T), x.T).T
  return x[:,:2]
  # return x

from torch.utils.data import Dataset, DataLoader

class toy_dataset(Dataset):
  def __init__(self, points):
    self.points = points
    print(len(points))

  def __len__(self):
    return len(self.points)
  
  def __getitem__(self,idx):
    return self.points[idx]


# dataset_transformed = transform(np.copy(dataset))
dataset_nonlinear = add_nonlinearity(np.copy(dataset))
dataset_nonlinear[:,0] = dataset_nonlinear[:,0]*1e3
dataset_nonlinear[:,1] = dataset_nonlinear[:,1]*1e7

toy_set = toy_dataset(dataset)

dataloader = DataLoader(toy_set, batch_size = args.batch_size, shuffle = True)

centers = [
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),
    (1. / np.sqrt(2), 1. / np.sqrt(2)),
    (1. / np.sqrt(2), -1. / np.sqrt(2)),
    (-1. / np.sqrt(2), 1. / np.sqrt(2)),
    (-1. / np.sqrt(2), -1. / np.sqrt(2))
  ]
centers2 = []
theta = np.pi/8
a = np.cos(theta)
b = np.sin(theta)
for p in centers:
  centers2.append((p[0]*a - p[1]*b, p[1]*a + p[0]*b))

scale = 2.0/1.414
centers2 = [(scale * x, scale * y) for x, y in centers2]

def f(x,y):
  sigma = 0.75
  val = None
  flag = 1
  for c in centers2:
    dis = torch.sqrt((x-c[0])**2 + (y-c[1])**2)
    if flag:
      flag = 0
      val = torch.tanh(dis*sigma).unsqueeze(0)
    else:
      val = torch.cat((val, torch.tanh(dis*sigma).unsqueeze(0)), dim = 0)
  #   val.append(np.tanh(dis*sigma))
  # print(val.size())
  # val = torch.tensor(val)
  # return 1 - torch.min(val, dim=0)[0]
  # return 1 - (np.sum(val, axis = 0)- 6.03)/1.97
  return 1 - (torch.sum(val,dim=0) - 5.11)/2.89

def train(gen, dis, gen_optim, dis_optim, args, experiment):
  gen.train()
  # dis.train()
  loss_func = torch.nn.BCELoss()

  experiment.train()
  n_critic = 1
  total_it = 40e3
  it = -1

  while(True):

    if it>= total_it:
      break

    for _, imgs in enumerate(dataloader):
      it+=1

      # if it%ewc_add == 0:
      #   break

      gen.train()
      # dis.train()
      for p in gen.parameters():
          gen.requires_grad = True
      # for p in dis.parameters():
      #     dis.requires_grad = True

      # dis_optim.zero_grad()
      
      # z = Variable(torch.rand(imgs.size()[0], args.Z_dim).cuda())
      ones = Variable(Tensor(imgs.size()[0], 1).fill_(1.0), requires_grad=False)
      # zeros = Variable(Tensor(imgs.size()[0], 1).fill_(0.0), requires_grad=False)

      # critic_fake = dis(gen(z))

      # # imgs = next(data)
      # # imgs = torch.tensor(imgs)
      # imgs = Variable(imgs.type(Tensor))

      # critic_real = dis(imgs)

      # critic_real_loss = loss_func(critic_real, ones)
      # critic_fake_loss = loss_func(critic_fake, zeros)
      # critic_loss = critic_real_loss + critic_fake_loss

      # critic_loss.backward()

      # # ewc
      # # ewc.update()
      # # loss_ewc = importance*ewc.penalty()
      # # if loss_ewc!=0:
      # #   loss_ewc.backward()


      # dis_optim.step()

      # weight clipping...

      # for p in dis.parameters():
      #   p.data.clamp_(-0.01, 0.01)

      # dis_optim.zero_grad()
        
      # generator
      if it%n_critic == 0:

        gen_optim.zero_grad()
        for p in dis.parameters():
          dis.requires_grad = False

        z = Variable(torch.rand(ones.size()[0], args.Z_dim).cuda())

        fakes = gen(z)
        critic_fake = f(fakes[:,0], fakes[:,1])

        gen_loss = -torch.mean(critic_fake)

        gen_loss.backward()
        gen_optim.step()


      if it%1000 == 0:

        # if loss_ewc!=0:
        #   print(loss_ewc.item())

        gen.eval()
        # dis.eval()
        for p in gen.parameters():
          gen.requires_grad = False

        print(it)
        print("gen_loss:{}".format(gen_loss.cpu().data.numpy()))
        print('------')

        experiment.log_metric("generator_loss", gen_loss.item())
        # experiment.log_metric("critic_loss", critic_loss.item())

        # plot

        z = Variable(torch.rand(args.batch_size, args.Z_dim).cuda())

        generated_points = gen(z).cpu().data.numpy()
        # print(generated_images[0])
        
        fig = plt.figure()

        ax=fig.add_subplot(111, label="1")
        # ax.set_xlim([-2,2])
        # ax.set_ylim([-2,2])
        ax2=fig.add_subplot(111, label="2", frame_on=False)

        for x in imgs.cpu().data.numpy():
          ax.scatter(x[0],x[1],c = 'g')

        for x in generated_points:
          ax.scatter(x[0],x[1], c = 'b')
        
        xp = np.linspace(-1.5,1.5,100)
        yp = np.linspace(-1.5,1.5,100)

        Dinput = []
        for i in range(len(xp)):
          for j in range(len(yp)):
            Dinput.append([xp[i],yp[j]])
        Dinput = np.array(Dinput)

        # Z = dis(torch.Tensor(add_nonlinearity(Dinput)).cuda())
        # # Z = dis(torch.Tensor(Dinput).cuda())
        X,Y = np.meshgrid(xp,yp)
        Z = f(torch.tensor(X), torch.tensor(Y))

        # ax = sns.heatmap(Z, alpha = 0.3).invert_yaxis()
        ax2.imshow(Z, cmap = 'plasma', aspect = 'auto', alpha = 0.8, origin='lower', vmin = 0.0, vmax = 0.99)
        
        ax.xaxis.tick_top()
        ax.yaxis.tick_right()
        
        experiment.log_figure(figure=plt, figure_name = "figure_" + str(it))
        # plt.show()
        plt.close()

# from comet_ml import Experiment
experiment = comet_ml.Experiment(project_name="prefixed_optimalD_wgan")

exp_parameters = {
    "data": "8gaussians_linear",
    "model": "WGAN_og_noBN",
    "opt_gen": "ADAM_lr_0.0001",
    "opt_dis": "ADAM_lr_0.0001",
    "z_dim": 10,
    "fixedD": "sum(5.11,2.89), tanh(/0.75)",
}

experiment.log_parameters(exp_parameters)

train(gen, dis, gen_optim, dis_optim, args, experiment)

