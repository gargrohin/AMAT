# import torch
# from torch import nn
# from torch.autograd import Variable
# from torch.nn import functional as F
# import torch.utils.data

# from torchvision.models.inception import inception_v3

# import numpy as np
# from scipy.stats import entropy



# # Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# import math
# import torch
# import torch.nn.functional as F


# class InceptionScore():
#     def __init__(self, classifier):

#         self.sumEntropy = 0
#         self.sumSoftMax = None
#         self.nItems = 0
#         self.classifier = classifier.eval()

#     def updateWithMiniBatch(self, ref):
#         y = self.classifier(ref).detach()

#         if self.sumSoftMax is None:
#             self.sumSoftMax = torch.zeros(y.size()[1]).to(ref.device)

#         # Entropy
#         x = F.softmax(y, dim=1) * F.log_softmax(y, dim=1)
#         self.sumEntropy += x.sum().item()

#         # Sum soft max
#         self.sumSoftMax += F.softmax(y, dim=1).sum(dim=0)

#         # N items
#         self.nItems += y.size()[0]

#     def getScore(self):

#         x = self.sumSoftMax
#         x = x * torch.log(x / self.nItems)
#         output = self.sumEntropy - (x.sum().item())
#         output /= self.nItems
#         return math.exp(output)
    
    
    
    
    

    
    
# def inception_score2(imgs, cuda=True, batch_size=32, resize=False, splits=10):

#     dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)
#     dtype = torch.cuda.FloatTensor
    
#     # Building the score instance
#     classifier = inception_v3(pretrained=True).cuda()
#     scoreMaker = InceptionScore(classifier)


#     print("Computing the inception score...")
#     up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
#     for i, batch in enumerate(dataloader, 0):
#         batchv = up(batch.cuda())
#         scoreMaker.updateWithMiniBatch(batchv)

#     print("Merging the results, please wait it can take some time...")
#     score = scoreMaker.getScore()

#     # Now printing the results
#     print(score)
        
        
        
        
# if __name__ == '__main__':
#     class IgnoreLabelDataset(torch.utils.data.Dataset):
#         def __init__(self, orig):
#             self.orig = orig

#         def __getitem__(self, index):
#             return self.orig[index][0]

#         def __len__(self):
#             return len(self.orig)

#     import torchvision.datasets as dset
#     import torchvision.transforms as transforms

#     cifar = dset.CIFAR10(root='data/', download=True, 
#                              transform=transforms.Compose([
#                                  transforms.Scale(32),
#                                  transforms.ToTensor(),
# #                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#                              ])
#     )

#     IgnoreLabelDataset(cifar)

#     print ("Calculating Inception Score...")
#     print (inception_score2(IgnoreLabelDataset(cifar), cuda=True, batch_size=50, resize=True, splits=10))






import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data

from torchvision.models.inception import inception_v3

import numpy as np
from scipy.stats import entropy

try:
    from torchvision.models.utils import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
    
from inception import fid_inception_v3
    
FID_WEIGHTS_URL = 'https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth'


def inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=1):
    """Computes the inception score of the generated images imgs

    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
#     inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
#     state_dict = torch.load("pt_inception-2015-12-05-6726825d.pth")
#     inception_model.load_state_dict(state_dict)
    inception_model = fid_inception_v3().type(dtype)
    
    
    inception_model.eval();
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x, dim=1).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1008))
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader, 0):
            batch = batch.type(dtype)
            batch_size_i = batch.size()[0]

            preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batch)

    # Now compute the mean kl-div
    split_scores = []

#     for k in range(splits):
#         part = preds[k * (N // splits): (k+1) * (N // splits), :]
#         py = np.mean(part, axis=0)
#         scores = []
#         for i in range(part.shape[0]):
#             pyx = part[i, :]
#             scores.append(entropy(pyx, py))
#         split_scores.append(np.exp(np.mean(scores)))


    for i in range(splits):
        part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
        kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        kl = np.mean(np.sum(kl, 1))
        split_scores.append(np.exp(kl))
            

    return np.mean(split_scores), np.std(split_scores)

if __name__ == '__main__':
    class IgnoreLabelDataset(torch.utils.data.Dataset):
        def __init__(self, orig):
            self.orig = orig

        def __getitem__(self, index):
            return self.orig[index][0]

        def __len__(self):
            return len(self.orig)

    import torchvision.datasets as dset
    import torchvision.transforms as transforms

    cifar = dset.CIFAR10(root='data/', download=True,
                             transform=transforms.Compose([
                                 transforms.Scale(32),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                             ])
    )

    IgnoreLabelDataset(cifar)

    print ("Calculating Inception Score...")
    print (inception_score(IgnoreLabelDataset(cifar), cuda=True, batch_size=50, resize=True, splits=10))
