import torchvision.models as models
from pytorch_lightning.metrics.functional import accuracy

from torchvision.io import read_image
from skimage import io
import numpy as np
import pandas as pd
import os
from PIL import Image
from sklearn.utils import shuffle
from glob import glob
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F

from time import time
import matplotlib.pyplot as plt
from sklearn import metrics
import pickle as pkl
import cv2

import pandas as pd
from pathlib import Path





class XrayDataset(Dataset):
    def __init__(self, df, transform = None, parent_dir = Path("mimic/mimic_scaled/")):
        self.annotations = df
        self.transform = transform
        self.parent_dir = parent_dir
    def __len__(self):
        return len(self.annotations)
    def __getitem__(self, index):
        img_id = self.annotations.iloc[index].img_id

        img_path = str(self.parent_dir/img_id)
        
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            print("image not found")
            # image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        
            # image = self.resize_to(image, 240)
            # cv2.imwrite( img_path[:-4]+ "_scaled.jpg" ,  image)
        
#         print(image.shape)
#         image = read_image(img_path)
        y_label = self.annotations.iloc[index].class_id
        if self.transform:
            image = self.transform(image)
        return [image, y_label]

    def resize_to(self, image, target_size = 240):
        resize_ratio = target_size/min(image.shape[:2])
        (h, w) = image.shape[:2]
        h_new = int(h*resize_ratio)
        w_new = int(w*resize_ratio)
        
        
        
        img = cv2.resize(image, (w_new,h_new), interpolation=cv2.INTER_AREA)
        
        
        return img