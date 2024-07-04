import os
import numpy as np
import copy as cp
import random as rd
import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
import time
import math
import scipy.stats as st
from collections import OrderedDict, defaultdict
import json

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18

from PIL import Image, ImageFilter
from sklearn.model_selection import train_test_split

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# Set random seed for reproducibility
seed = 1234
np.random.seed(seed)
rd.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

# Configurations
device = "cuda"
base_root = ""
symbol = "//"
model_root = base_root
