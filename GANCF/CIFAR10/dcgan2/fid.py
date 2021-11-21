import os
import pathlib
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import numpy as np
import torch
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d

from PIL import Image

try:
    from tqdm import tqdm
except ImportError:
    # If not tqdm is not available, provide a mock version of it
    def tqdm(x): return x

from inception import InceptionV3







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





def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)





def get_fid_activations(imgs, model, batch_size=50, dims=2048, cuda=True):
    
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

#     # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
#     inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
#     state_dict = torch.load("pt_inception-2015-12-05-6726825d.pth")
#     inception_model.load_state_dict(state_dict)
#     inception_model = fid_inception_v3().type(dtype)
    
    
    model.eval();
    
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    
    def get_pred(x):
        x = up(x)
        x = model(x)[0]
        if x.size(2) != 1 or x.size(3) != 1:
            x = adaptive_avg_pool2d(pred, output_size=(1, 1))
            
        return x.cpu().data.numpy().reshape(x.size(0), -1)

    # Get predictions
    preds = np.zeros((N, 2048))
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader, 0):
            batch = batch.type(dtype)
            batch_size_i = batch.size()[0]

            preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batch)


    return preds

  

def calculate_activation_statistics(files, model, batch_size=50, dims=2048,
                                    cuda=True):
    """Calculation of the statistics used by the FID.
    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    act = get_fid_activations(files, model, batch_size, dims, cuda)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma



def get_fid_score(G, real_data_loader):
    
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    model = InceptionV3([block_idx]).cuda()
    
    batch_size = 50
    dims = 2048
    cuda = True
    
    
    G.eval()
    
    batch_size = 100
    with torch.no_grad():
        for i in range(100):
            z = Variable(torch.randn(batch_size, 100, 1, 1).cuda())
            img = G(z)
            if i == 0:
                images = img
            else:
                images = torch.cat((images, img), dim = 0)
                
    
    with torch.no_grad():
        for i, (batch, _) in enumerate(real_data_loader, 0):
            
            if i == 0:
                images_real = batch
            else:
                images_real = torch.cat((images_real, batch), dim = 0)
                
#             if(images_real.size(0)>10000):
#                 break
    
    m1, s1 = calculate_activation_statistics(images, model, batch_size,
                                             dims, cuda)
    m2, s2 = calculate_activation_statistics(images_real, model, batch_size,
                                     dims, cuda)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)
    
    return fid_value