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

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device: ", device)

bs = 100

# MNIST Dataset
transform = transforms.Compose([
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

class Generator(nn.Module):
    def __init__(self, g_input_dim, g_output_dim):
        super(Generator, self).__init__()       
        self.fc1 = nn.Linear(g_input_dim, 256)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features*2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features*2)
        self.fc4 = nn.Linear(self.fc3.out_features, g_output_dim)
    
    # forward method
    def forward(self, x): 
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        return torch.tanh(self.fc4(x))
    
class Discriminator(nn.Module):
    def __init__(self, d_input_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(d_input_dim, 1024)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features//2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features//2)
        self.fc4 = nn.Linear(self.fc3.out_features, 1)
    
    # forward method
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = F.dropout(x, 0.3)
        return torch.sigmoid(self.fc4(x))

# build network
z_dim = 100
mnist_dim = train_dataset.data.size(1) * train_dataset.data.size(2)

G = Generator(g_input_dim = z_dim, g_output_dim = mnist_dim).to(device)
D = Discriminator(mnist_dim).to(device)

# loss
criterion = nn.BCELoss() 

# optimizer
lr = 0.0002

G_optimizer = optim.Adam(G.parameters(), lr = lr, betas = (0.5, 0.999))
D_optimizer = optim.Adam(D.parameters(), lr = lr, betas = (0.5, 0.999))

#G_optimizer = optim.SGD(G.parameters(), lr = lr)
#D_optimizer = optim.SGD(D.parameters(), lr = lr)

def D_train(x):
    #=======================Train the discriminator=======================#
    D.zero_grad()

    # train discriminator on real
    x_real, y_real = x.view(-1, mnist_dim), torch.ones(bs, 1)
    x_real, y_real = Variable(x_real.to(device)), Variable(y_real.to(device))

    D_output_real = D(x_real)
    # D_real_loss = criterion(D_output_real, y_real)
    D_real_score = D_output_real

    # train discriminator on facke
    z = Variable(torch.randn(bs, z_dim).to(device))
    x_fake, y_fake = G(z), Variable(torch.zeros(bs, 1).to(device))

    D_output_fake = D(x_fake)
    # D_fake_loss = criterion(D_output_fake, y_fake)
    D_fake_score = D_output_fake

    D_acc = get_critic_acc(D_output_fake, D_output_real)
    D_loss = -(torch.mean(D_output_real) - torch.mean(D_output_fake))

    # gradient backprop & optimize ONLY D's parameters
    # D_loss = D_real_loss + D_fake_loss
    D_loss.backward()
    D_optimizer.step()

    # weight clipping
    for p in D.parameters():
        p.data.clamp_(-0.01, 0.01)
        
    return  D_loss.data.item(), D_acc

def G_train(x):
    #=======================Train the generator=======================#
    G.zero_grad()

    z = Variable(torch.randn(bs, z_dim).to(device))
    y = Variable(torch.ones(bs, 1).to(device))

    G_output = G(z)
    D_output = D(G_output)
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

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

experiment = comet_ml.Experiment(project_name="mnist_baselines_Wgan")

exp_parameters = {
    "data": "mnist_28x28",
    "model": "WGAN_4fc_dropout_dis",
    "opt_gen": "ADAM_lr_0.0002, betas = (0.5,0.999)",
    "opt_dis": "ADAM_lr_0.0002, betas = (0.5,0.999)",
    "z_dim": 100,
    "normalize": "mean:0.5, std:0.5",
    "dis_landscape": 1,
    "try" : 0
}

experiment.log_parameters(exp_parameters)

experiment.train()

n_epoch = 50
gen_samples = []

for epoch in range(1, n_epoch+1):
    G.train()         
    D_losses, D_accuracy, G_losses = [], [], []
    for batch_idx, (x, _) in enumerate(train_loader):
        D_loss, D_acc = D_train(x)
        D_losses.append(D_loss)
        D_accuracy.append(D_acc)
        G_losses.append(G_train(x))

    print('[%d/%d]: loss_d: %.3f, acc_d: %.3f, loss_g: %.3f' % (
            (epoch), n_epoch, torch.mean(torch.FloatTensor(D_losses)), torch.mean(torch.FloatTensor(D_accuracy)), torch.mean(torch.FloatTensor(G_losses))))
    
    with torch.no_grad():
        G.eval()
        test_z = Variable(torch.randn(bs, z_dim).to(device))
        generated = G(test_z)
        gen_samples.append(generated)
        generated = generated.cpu().data.numpy()[:64]

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
            plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

        experiment.log_figure(figure=plt, figure_name = "figure_" + str(epoch))
        plt.close()     
        
# Sample data to get discriminator landscape.
samples = []
done = []
for i in range(10):
    done.append(0)
for x,y in train_loader:
    if done[y[0]] == 0:
        samples.append(x[0])
        done[y[0]]=1

print()

D_outs = []
D_outs_fake = []

for img in samples:
    D_outs.append(torch.mean(D(img.view(-1, mnist_dim).float().cuda())).cpu().detach().numpy())
for img in gen_samples[10:]:
    D_outs_fake.append(torch.mean(D(img.view(-1, mnist_dim).float().cuda())).cpu().detach().numpy())

plt.ylim((0,1))
for i in range(10):
    plt.scatter(i,D_outs[i], c = 'g')
    print(i , D_outs[i])
plt.title("D_modes")
experiment.log_figure(figure=plt, figure_name = "dis_modes")
plt.close()

print()

#gen samples
for j in range(10, len(gen_samples)):
  generated = gen_samples[j]
  generated = generated.cpu().data.numpy()[:4]

  fig = plt.figure(figsize=(2, 2))
  gs = gridspec.GridSpec(2, 2)
  gs.update(wspace=0.05, hspace=0.05)

  for i, sample in enumerate(generated):
    ax = plt.subplot(gs[i])
    plt.axis('off')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_aspect('equal')
    plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

#   plt.title("gen_sample_" + str(j) + "_" + str(D_outs_fake[j-10]))
  experiment.log_figure(figure=plt, figure_name = "gen_sample_" + str(j-10))
  print(j , D_outs_fake[j-10])

plt.close()
print()
plt.ylim((0,1))

for i in range(len(D_outs_fake)):
  plt.scatter(i,D_outs_fake[i], c = 'g')
plt.title("gen_samples_finalD_score")
experiment.log_figure(figure=plt, figure_name = "gen_samples_finalD_score")

plt.close()
