import os
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import make_grid
import torch.nn.functional as F
from imageio import imsave
from tqdm import tqdm
from copy import deepcopy
import logging
from torch.autograd import Variable
import cfg
import models
import datasets
from functions import train_multi, validate, LinearLrDecay, load_params, copy_params
from utils.utils import set_log_dir, save_checkpoint, create_logger
from utils.inception_score import _init_inception
from utils.fid_score import create_inception_graph, check_or_download_inception

from utils.inception_score import get_inception_score
from utils.fid_score import calculate_fid_given_paths

_init_inception()
inception_path = check_or_download_inception(None)
create_inception_graph(inception_path)

fid_buffer_dir = 'data/CUB_200_2011/all_images/'
#os.makedirs(fid_buffer_dir)

# fname = '.samples.npz'
# print('loading %s ...'%fname)
# ims = np.load(fname)['x']
# ims = list(ims.swapaxes(1,2).swapaxes(2,3))
# mean, std = get_inception_score(ims)
# print(mean,std)
fid_stat = 'fid_stat/fid_stats_cifar10_train.npz'
fid_score = calculate_fid_given_paths([fid_buffer_dir, fid_stat], inception_path=None)
print(fid_score)

