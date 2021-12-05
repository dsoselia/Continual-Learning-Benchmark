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
from tqdm import tqdm





class XrayDataset(Dataset):
    def __init__(self, df, transform = None, parent_dir = Path("data/mimic"), preload = True): #
        self.annotations = df
        self.transform = transform
        self.parent_dir = parent_dir
        self.root = parent_dir
        self.preload = preload

        if preload:
            X = []
            y = []
            print("loading Xray data")
            for i, row in tqdm(df.iterrows(), total = len(df)):
                img_id = row.img_id
                img_path = str(self.parent_dir/"mimic_scaled"/img_id)
                image = cv2.imread(img_path, cv2.IMREAD_COLOR)
                y_label = row.class_id
                X.append(image)
                y.append(y_label)
            self.X = X
            self.y = np.array(y)

    def __len__(self):
        return len(self.annotations)
    def __getitem__(self, index):
        y_df = self.annotations.iloc[index].class_id

        if self.preload:
            image = self.X[index]
            y_label = self.y[index]
            if y_label != y_df:
                print("mismatch")
                raise ValueError
        else:
            img_id = self.annotations.iloc[index].img_id
            img_path = str(self.parent_dir/"mimic_scaled"/img_id)
            
            image = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if image is None:
                print("image not found")
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



class XrayDatasetExtended(Dataset):
    def __init__(self, df, transform = None, parent_dir = Path("data/mimic"), preload = True): #
        self.annotations = df
        self.transform = transform
        self.parent_dir = parent_dir
        self.root = parent_dir
        self.preload = preload

        if preload:
            X = []
            y = []
            print("loading Xray data")
            for i, row in tqdm(df.iterrows(), total = len(df)):
                img_id = row.img_id
                img_path = str(self.parent_dir/"mimic_scaled"/img_id)
                image = cv2.imread(img_path, cv2.IMREAD_COLOR)
                y_label = row.continual_class_id
                X.append(image)
                y.append(y_label)
            self.X = X
            self.y = np.array(y)

    def __len__(self):
        return len(self.annotations)
    def __getitem__(self, index):
        y_df = self.annotations.iloc[index].continual_class_id

        if self.preload:
            image = self.X[index]
            y_label = self.y[index]
            if y_label != y_df:
                print("mismatch")
                raise ValueError
        else:
            img_id = self.annotations.iloc[index].img_id
            img_path = str(self.parent_dir/"mimic_scaled"/img_id)
            
            image = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if image is None:
                print("image not found")
            y_label = self.annotations.iloc[index].continual_class_id
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