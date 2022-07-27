import pathlib
import torch
from torch.utils.data import Dataset
import glob
import os
from tifffile import TiffFile
import numpy as np

class OneClassDataset(Dataset):
    
    def __init__(self, config, class_name, is_training):
        self.class_name = class_name
        self.config = config
        if (is_training):
            self.dataset_path = self.config.dataset.train_dataset_path
        else:
            self.dataset_path = self.config.dataset.test_dataset_path
        
        self.file_list = glob.glob(self.dataset_path + class_name + "/*.tif")

    def __getitem__(self, index):
        filepath = self.file_list[index]
        img = TiffFile(filepath).asarray().astype(np.float32)
        label = self.config.model.class_names.index(self.class_name)
        img_n_RGB = img[:, :, :3][..., ::-1]  # 2 -> R, 1 -> G, 0 -> B
        img_n_NIRG = img[:, :, 1:][..., ::-1]  # 3-> N, 2-> R, 1-> G  
        img_n_NI = img[:, :, 3][:,:,None]
        return img_n_RGB,img_n_NIRG,img_n_NI,label

    def __len__(self):
        return len(self.file_list)
