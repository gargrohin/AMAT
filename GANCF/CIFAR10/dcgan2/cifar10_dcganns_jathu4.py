import wandb

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
import random
from tqdm import tqdm

import torchvision.utils as vutils
from torchvision.utils import save_image
import pickle
import argparse
from scipy.stats import entropy
from inception_score import inception_score



from models import Discriminator, Generator, weights_init_normal
from utils import inception_eval, inception_eval_cifar10
from learner import Learner #D_train, G_train, get_critic_acc
from fid import get_fid_score



import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec



def parse_option():

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--eval_freq', type=int, default=5, help='meta-eval frequency')
    parser.add_argument('--save_freq', type=int, default=10, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=200, help='number of training epochs')
    parser.add_argument('--z_dim', type=int, default=100, help='embedding dimension for transformer')
    parser.add_argument('--n_dis', type=int, default=4, help='number of discrimnators')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.0002, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='15', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.0, help='decay rate for learning rate')
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





# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device: ", device)



opt = parse_option()
wandb.init(project=opt.model_path.split("/")[-1], tags=opt.tags)
wandb.config.update(opt)
wandb.save('*.py')
try:
    path = os.path.join(wandb.run.dir, "codes")
    os.system('mkdir '+ path)
    os.system('cp *.py '+ path)
except:
    pass
    
wandb.run.save()

    

# CIFAR10 Dataset
transform = transforms.Compose([
    transforms.Scale(64),
    transforms.ToTensor(),
#    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))])

dataset = datasets.CIFAR10(root='../datasets/cifar10_data/', transform=transform, download=True)
dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)



n_epoch = 200
n_critic = 2
bs = 128
output = '.temp_2.png'
is_score, is_std, fid_score = 0, 0, 1000
best_is_score = 0
learner = Learner(opt, opt.z_dim, opt.batch_size, device)

# wandb.watch(learner.G)
# wandb.watch(learner.D)

print(learner.G)
print(learner.multiD[0])

# inception_eval_cifar10()

for epoch in range(1, opt.epochs+1):

#     if (epoch in opt.lr_decay_epochs):
#         steps = np.sum(epoch >= np.asarray(opt.lr_decay_epochs))
#         print(steps)
#         learner.adjust_learning_rate(0.1**steps)
    

    learner.G.train()
#     learner.D.train()
    D_losses, D_accuracy, G_losses = [], [], []
    

            
    with tqdm(dataloader, total=len(dataloader)) as pbar:
        for batch_idx, (x, _) in enumerate(pbar):
            G_loss = learner.G_train(x)
            G_losses.append(G_loss)

            if batch_idx % n_critic == 0:
                D_loss, D_acc, _ = learner.D_train(x)
                D_losses.append(D_loss)
                D_accuracy.append(D_acc)
            
    print(wandb.run.name)
    print('[%d/%d]: loss_d: %.3f, acc_d: %.3f, loss_g: %.3f' % ((epoch), opt.epochs, torch.mean(torch.FloatTensor(D_losses)), torch.mean(torch.FloatTensor(D_accuracy)), torch.mean(torch.FloatTensor(G_losses))))
    
    if epoch%opt.save_freq == 0:        
        print('==> Saving...')
        state = {
            'epoch': epoch,
            'optimizer_G': learner.G_optimizer.state_dict(),
#             'optimizer_D': learner.D_optimizer.state_dict(),
            'model_G': learner.G.state_dict(),
#             'model_D': learner.D.state_dict(),
        }            
        save_file = os.path.join(opt.save_folder, 'model_'+str(wandb.run.name)+'.pth')
        torch.save(state, save_file)

        #wandb saving
        torch.save(state, os.path.join(wandb.run.dir, "model.pth"))
            
            
    if epoch%opt.eval_freq == 0:
        with torch.no_grad():
            learner.G.eval()
            test_z = Variable(torch.randn(64, opt.z_dim, 1, 1).to(device))
            generated = learner.G(test_z).detach().cpu()

    #         vutils.save_image(generated, output ,normalize=True)

            
            if epoch%opt.eval_freq == 0:
                is_score, is_std = inception_eval(learner.G, device, opt, 100)
                
    #             fid_score = 0
    #             if(is_score>5.5):
                fid_score = get_fid_score(learner.G, dataloader)
                print("XXXXXX", fid_score)
        
            if(best_is_score<is_score):
                best_is_score = is_score
                
                
                state = {
                    'epoch': epoch,
                    'optimizer_G': learner.G_optimizer.state_dict(),
    #                 'optimizer_D': learner.D_optimizer.state_dict(),
                    'model_G': learner.G.state_dict(),
    #                 'model_D': learner.D.state_dict(),
                }            
                save_file = os.path.join(opt.save_folder, 'best_model_'+str(wandb.run.name)+'.pth')
                torch.save(state, save_file)

                #wandb saving
                torch.save(state, os.path.join(wandb.run.dir, "best_model.pth"))
            
            
                
        
        wandb.log({'epoch': epoch, 
                'critic_loss': torch.mean(torch.FloatTensor(D_losses)),
                'gen_loss':torch.mean(torch.FloatTensor(G_losses)),
                'critic_acc': torch.mean(torch.FloatTensor(D_accuracy)),
                'Inception Score':is_score,
                'Inception std':is_std,
                'Fid Score':fid_score,
                }, commit=True)
                
