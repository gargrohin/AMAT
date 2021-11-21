import comet_ml
comet_ml.config.save(api_key="CX4nLhknze90b8yiN2WMZs9Vw")

import os
os.environ["CUDA_VISIBLE_DEVICES"]='0,1,2,3,4,5,6,7,8,9'


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torch.autograd import Variable
from copy import deepcopy

batch_size = 100

import random

def gaussians():  
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
  centers1 = [(scale * x, scale * y) for x, y in centers1]
  while True:
    dataset = []
    for i in range(batch_size):
      point = np.random.randn(2) * 0.05
      center = random.choice(centers1)
      point[0] += center[0]
      point[1] += center[1]
      dataset.append(point)
    dataset = np.array(dataset, dtype='float32')
    dataset /= 1.414  # stdev
    yield dataset


data = gaussians()
points = next(data)

all_points = np.copy(points)
dataset = np.copy(points)

cuda = True if torch.cuda.is_available() else False
# print("cuda " , cuda)
device = torch.device("cuda:7")
print(device)

import torch.nn.functional as F
from torch.autograd import Variable

class Wgen(nn.Module):
  
  def __init__(self, args):
    super(Wgen, self).__init__()

    self.model = nn.Sequential(
                 nn.Linear(args.Z_dim, args.dim1),
                #  nn.BatchNorm1d(args.dim1, 0.8),
                 nn.ReLU(),
                 nn.Linear(args.dim1, args.dim2),
                #  nn.BatchNorm1d(args.dim2, 0.8),
                 nn.ReLU(),
                 nn.Linear(args.dim2, args.dim3),
                #  nn.BatchNorm1d(args.dim3, 0.8),
                 nn.ReLU(),
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
    # print(x)
    # x = x.view(x.shape[0], -1)
    # print(x)
    out = self.model(x)
    return out

import argparse

parser = argparse.ArgumentParser(description='CNN text classificer')
# learning
parser.add_argument('-lr', type=float, default=0.01, help='initial learning rate [default: 0.001]')
parser.add_argument('-epochs', type=int, default=300, help='number of epochs for train [default: 256]')
parser.add_argument('-batch_size', type=int, default=128, help='batch size for training [default: 64]')
parser.add_argument('-Z_dim', type=int, default = 100)
parser.add_argument('-dim1', type=int, default = 128)
parser.add_argument('-dim2', type=int, default = 512)
parser.add_argument('-dim3', type=int, default = 1024)
parser.add_argument('-out_dim', type=int, default = 1000)
parser.add_argument('-img_dim', type=int, default = 1000)
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
  gen.to(device)
  dis.to(device)

# gen_optim = optim.RMSprop(gen.parameters(), lr = args.lr)
# dis_optim = optim.RMSprop(dis.parameters(), lr = args.lr)

gen_optim = optim.Adam(gen.parameters(), lr = args.lr*0.01, betas = (0.0,0.9))
# dis_optim = optim.Adam(dis.parameters(), lr = args.lr, betas = (0.0,0.9))

# gen_optim = optim.SGD(gen.parameters(), lr=args.lr, momentum=0.0)
dis_optim = optim.SGD(dis.parameters(), lr=args.lr, momentum=0.0)


# Transform matrix

A = (np.random.rand(args.img_dim, 2+1) -0.5 )*2  # keep A values in [-1,1]


def transform(x):
  # x = add_nonlinearity(x)
  x = np.vstack((x.T, np.ones(x.shape[0])))
  B = np.dot(A, x)
  B = B + np.random.normal(0 , 10**(-3) , B.shape)
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
    # print(len(points))

  def __len__(self):
    return len(self.points)
  
  def __getitem__(self,idx):
    return self.points[idx]


dataset_transformed = transform(np.copy(dataset))
# dataset_nonlinear = add_nonlinearity(np.copy(dataset))
# print(dataset_nonlinear[:10])
toy_set = toy_dataset(dataset_transformed)

dataloader = DataLoader(toy_set, batch_size = args.batch_size, shuffle = True)

EPS = 1e-20
def normalize_fn(fisher):
    return (fisher - fisher.min()) / (fisher.max() - fisher.min() + EPS)

class EWCpp(object):
    def __init__(self, model, model_old, device, alpha=0.9, fisher=None, normalize=True):

        self.model = model
        self.model_old = model_old
        self.model_old_dict = self.model_old.state_dict()

        self.device = device
        self.alpha = alpha
        self.normalize = normalize
        
        if fisher is not None: # initialize as old Fisher Matrix
            self.fisher_old = fisher
            for key in self.fisher_old:
                self.fisher_old[key].requires_grad = False
                self.fisher_old[key] = self.fisher_old[key].to(device)
            self.fisher = deepcopy(fisher)
            if normalize:
                self.fisher_old = {n: normalize_fn(self.fisher_old[n]) for n in self.fisher_old}

        else: # initialize a new Fisher Matrix
            self.fisher_old = None
            self.fisher = {n:torch.zeros_like(p, device=device, requires_grad=False) 
                           for n, p in self.model.named_parameters() if p.requires_grad} 

    def update(self):
        # suppose model have already grad computed, so we can directly update the fisher by getting model.parameters
        for n, p in self.model.named_parameters():
            if p.grad is not None:
                self.fisher[n] = (self.alpha * p.grad.data.pow(2)) + ((1-self.alpha)*self.fisher[n])

    def get_fisher(self):
        return self.fisher # return the new Fisher matrix

    def penalty(self):
        loss = 0
        if self.fisher_old is None:
            return 0.
        for n, p in self.model.named_parameters():
            loss += (self.fisher_old[n] * (p - self.model_old_dict[n]).pow(2)).sum()
        return loss


def train(gen, dis, gen_optim, dis_optim, args, experiment):
  gen.train()
  dis.train()

  experiment.train()
  n_critic = 5
  total_it = 200000
  it = -1

  start_ewc = 3000
  ewc_add = 101
  # stop_ewc = ewc_add*20 + start_ewc

  importance = 5000
  fisher = None

  dis_old = deepcopy(dis)
  for p in dis_old.parameters():
    p.requires_grad = False

  ewc = EWCpp(dis, dis_old, device, fisher = fisher)

  while(True):


    if it>= total_it:
      break

    for _, imgs in enumerate(dataloader):
      it+=1

      if it > start_ewc:
        if it%ewc_add == 0:

          dis_old = deepcopy(dis)
          for p in dis_old.parameters():
            p.requires_grad = False
          
          fisher = deepcopy(ewc.get_fisher())
          ewc = EWCpp(dis, dis_old, device, fisher = fisher)

      gen.train()
      dis.train()
      for p in gen.parameters():
        p.requires_grad = True
      for p in dis.parameters():
        p.requires_grad = True

      dis_optim.zero_grad()
      
      z = Variable(torch.rand(args.batch_size, args.Z_dim).to(device))

      critic_fake = dis(gen(z))

      # imgs = next(data)
      # imgs = torch.tensor(imgs)
      imgs = Variable(imgs.type(Tensor))
      imgs = imgs.to(device)

      critic_real = dis(imgs)

      critic_loss = -(torch.mean(critic_real) - torch.mean(critic_fake))

      critic_loss.backward()

      # ewc
      ewc.update()
      loss_ewc = importance*ewc.penalty()
      if loss_ewc!=0:
        loss_ewc.backward()


      dis_optim.step()

      # weight clipping...

      for p in dis.parameters():
        p.data.clamp_(-0.01, 0.01)

      # dis_optim.zero_grad()
        
      # generator
      if it%n_critic == 0:

        gen_optim.zero_grad()
        for p in dis.parameters():
          dis.requires_grad = False

        z = Variable(torch.rand(args.batch_size, args.Z_dim).to(device))

        critic_fake = dis(gen(z))

        gen_loss = -torch.mean(critic_fake)

        gen_loss.backward()
        gen_optim.step()


      if it%200 == 0:
        if loss_ewc != 0.0:
          print(loss_ewc.item())
          experiment.log_metric("loss_ewc", loss_ewc.item())
        else:
          print(0.0)
          experiment.log_metric("loss_ewc", loss_ewc)

        gen.eval()
        dis.eval()
        for p in gen.parameters():
          gen.requires_grad = False

        print(it)
        print("gen_loss:{}, dis_loss: {}".format(gen_loss.cpu().data.numpy(), critic_loss.cpu().data.numpy()))
        print('------')

        experiment.log_metric("generator_loss", gen_loss.item())
        experiment.log_metric("critic_loss", critic_loss.item())

        # plot

        z = Variable(torch.rand(args.batch_size, args.Z_dim).to(device))

        generated_points = gen(z).cpu().data.numpy()
        # print(generated_images[0])
        
        fig = plt.figure()

        ax=fig.add_subplot(111, label="1")
        # ax.set_xlim([-2,2])
        # ax.set_ylim([-2,2])
        ax2=fig.add_subplot(111, label="2", frame_on=False)

        for x in all_points:
          ax.scatter(x[0],x[1],c = 'g')

        for x in detransform(generated_points):
          ax.scatter(x[0],x[1], c = 'b')
        
        xp = np.linspace(-1.5,1.5,100)
        yp = np.linspace(-1.5,1.5,100)

        Dinput = []
        for i in range(len(xp)):
          for j in range(len(yp)):
            Dinput.append([xp[i],yp[j]])
        Dinput = np.array(Dinput)

        Z = dis(torch.Tensor(transform(Dinput)).to(device))
        # Z = dis(torch.Tensor(Dinput).cuda())


        Z = Z.cpu().detach().numpy().reshape((len(xp),len(yp)))

        # ax = sns.heatmap(Z, alpha = 0.3).invert_yaxis()
        ax2.imshow(Z, cmap = 'plasma', aspect = 'auto', alpha = 0.5, origin='lower', vmin = 0.0, vmax = 0.99)
        
        ax.xaxis.tick_top()
        ax.yaxis.tick_right()
        
        experiment.log_figure(figure=plt, figure_name = "figure_" + str(it))
        plt.close(fig)


experiment = comet_ml.Experiment(project_name="ewc++_wgan_highdim")
print("experiment started...")
exp_parameters = {
    "data": "8gaussians_uniform_fixeddata_linear",
    "model": "WGAN_gen1BN_3layers",
    "opt_gen": "ADAM_lr_0.0001",
    "opt_dis": "SGD_lr_0.01",
    "z_dim": 100,
    "high_dim": 1000,
    "n_critic": 5,
    "dataset_size": 100,
    "noise_varience": "10^(-3)",
    "importance": 5000,
    "start_ewc": 5000,
    "ewc_add": 101,
    "alpha_ewc": 0.9
}
experiment.log_parameters(exp_parameters)
print("parameters logged...")

print("\nStart Training\n")

train(gen, dis, gen_optim, dis_optim, args,experiment)
