import pathlib
import torch
from torch.utils.data import Dataset
from model_training.one_class_dataset import OneClassDataset

class DatasetCreater:
    def __init__(self, config):
        self.config = config        
                    
    def create_train_val_dataset(self):

        train_dataset = torch.utils.data.ConcatDataset([
                OneClassDataset( self.config, class_name, True)
                for class_name in self.config.model.class_names
        ])
        
            
        val_ratio = self.config.dataset.val_ratio
        val_length = int(len(train_dataset)*val_ratio)
        train_length = len(train_dataset) - val_length
        length_list = [train_length, val_length]
        return torch.utils.data.dataset.random_split(train_dataset, length_list)

    def create_test_dataset(self):
        test_dataset = torch.utils.data.ConcatDataset([
                OneClassDataset( self.config, class_name, False)
                for class_name in self.config.model.class_names
        ])
                                    
        return test_dataset
