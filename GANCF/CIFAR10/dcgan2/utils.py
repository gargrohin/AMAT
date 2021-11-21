import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable

import numpy as np
from scipy.stats import entropy
from inception_score import inception_score

def inception_eval(G, device, opt, n_samples):
    
    G.eval()
    images_gan = []
    
    batch_size = 200
    with torch.no_grad():
        for i in range(n_samples):
            z = Variable(torch.randn(batch_size, opt.z_dim, 1, 1).to(device))
            img = G(z).cpu()
            if i == 0:
                images = img
            else:
                images = torch.cat((images, img), dim = 0)
#     print(images.shape)
#     images = images.detach().cpu().numpy()
#     print(images.shape)

    # important!
#     torch.cuda.empty_cache()

    incept = inception_score(images, cuda=True, batch_size=50, resize=True, splits=10)
    print("Inception score : ", incept)
#     experiment.log_metric("inception_score", incept[0])
    
    return incept

    
    
def inception_eval_cifar10():
    
    
    import torchvision.datasets as dset
    import torchvision.transforms as transforms

    cifar = dset.CIFAR10(root='data/', download=True, 
                             transform=transforms.Compose([
                                 transforms.Scale(32),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                             ])
    )    

    train_x = cifar.data

    train_x = np.moveaxis(train_x, -1, 1)
    train_x = (train_x-127.5)/127.5
    train_x = list(train_x)
    random.shuffle(train_x)
    
    print("\nCalculating IS...")
    incept = inception_score(train_x, cuda=True, batch_size=50, resize=True, splits=10)
    print(incept) 