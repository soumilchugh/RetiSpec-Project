import yacs.config
import cv2
from torch.utils.data import DataLoader
import torchvision
from model_training.transforms import Transform
from model_training.dataset_creater import DatasetCreater
import numpy as np
import torch


class DatasetLoader:

    def __init__(self, config, datasetCreater):
        self.config = config
        self.train_dataset = None
        self.val_dataset = None
        self.datasetCreater = datasetCreater
        self.device = self.config.model.device
        self.transform = Transform(config).create_transform()

    def collate_batch(self,batch):
        img_rgb_list, img_nirgb_list, img_nir_list, label_list = [], [], [],[]

        for (img_rgb, img_nirgb, img_nir, label) in batch:
            transformed = self.transform(image=img_rgb, image0=img_nirgb, image1=img_nir)
            
            img_rgb = transformed['image']
            img_rgb = img_rgb.transpose(2,0,1).astype(np.float32)
            torch_img_rgb = torch.from_numpy(img_rgb)

            img_nirgb = transformed['image0']
            img_nirgb = img_nirgb.transpose(2,0,1).astype(np.float32)
            torch_img_nirgb = torch.from_numpy(img_nirgb)

            img_nir = transformed['image1']
            img_nir = img_nir.transpose(2,0,1).astype(np.float32)
            torch_img_nirgb = torch.from_numpy(img_nir)
            
            img_rgb_list.append(torch_img_rgb)
            img_nirgb_list.append(torch_img_nirgb)
            img_nir_list.append(torch_img_nirgb)

            torch_label = torch.tensor(label, dtype=torch.float)
            torch_label = torch_label.unsqueeze(0)
            label_list.append(torch_label)

        images_rgb = torch.stack(img_rgb_list, dim=0)
        imgs_nirgb = torch.stack(img_nirgb_list, dim=0)
        imgs_nir = torch.stack(img_nir_list, dim=0)
        labels = torch.stack(label_list, dim=0)

        return images_rgb.to(self.device), imgs_nirgb.to(self.device), imgs_nir.to(self.device), labels.to(self.device)


    def collate_test_batch(self,batch):
        img_rgb_list, img_nirgb_list, img_nir_list, label_list = [], [], [], []

        for (img_rgb, img_nirgb, img_nir, label) in batch:
            img_rgb = img_rgb/65536
            img_nirgb = img_nirgb/65536
            img_nir = img_nir/65536
            
            img_rgb = img_rgb.transpose(2,0,1).astype(np.float32)
            torch_img_rgb = torch.from_numpy(img_rgb)

            img_nirgb = img_nirgb.transpose(2,0,1).astype(np.float32)
            torch_img_nirgb = torch.from_numpy(img_nirgb)
            
            img_nir = img_nir.transpose(2,0,1).astype(np.float32)
            torch_img_nirgb = torch.from_numpy(img_nir)
            
            img_rgb_list.append(torch_img_rgb)
            img_nirgb_list.append(torch_img_nirgb)
            img_nir_list.append(torch_img_nirgb)

            torch_label = torch.tensor(label,dtype=torch.float)
            torch_label = torch_label.unsqueeze(0)
            label_list.append(torch_label)

        images_rgb = torch.stack(img_rgb_list, dim=0)
        imgs_nirgb = torch.stack(img_nirgb_list, dim=0)
        imgs_nir = torch.stack(img_nir_list, dim=0)
        labels = torch.stack(label_list, dim=0)

        return images_rgb.to(self.device), imgs_nirgb.to(self.device), imgs_nir.to(self.device), labels.to(self.device)


    def load_train_val_data(self):
        train_dataset, val_dataset = self.datasetCreater.create_train_val_dataset()
        print ("Length of Train, Val Dataset ",len(train_dataset), len(val_dataset))

        train_loader = DataLoader (
                train_dataset,
                collate_fn = self.collate_batch,
                batch_size = self.config.hyperparameters.batch_size,
                shuffle = self.config.hyperparameters.shuffle,
                num_workers = self.config.hyperparameters.num_workers,
                pin_memory = self.config.hyperparameters.pin_memory,
                drop_last = self.config.hyperparameters.drop_last
            )
        
        val_loader = DataLoader (
                val_dataset,
                collate_fn = self.collate_test_batch,
                batch_size = self.config.hyperparameters.batch_size,
                shuffle = self.config.hyperparameters.shuffle,
                num_workers = self.config.hyperparameters.num_workers,
                pin_memory = self.config.hyperparameters.pin_memory,
                drop_last = self.config.hyperparameters.drop_last
            )
        
        return train_loader, val_loader

    def load_test_data(self):
        
        test_dataset = self.datasetCreater.create_test_dataset()
        print ("Length of Test Dataset ",len(test_dataset))

        test_loader = DataLoader(
                test_dataset,
                collate_fn = self.collate_test_batch,
                batch_size = self.config.hyperparameters.batch_size,
                shuffle = self.config.hyperparameters.shuffle,
                num_workers = self.config.hyperparameters.num_workers,
                pin_memory = self.config.hyperparameters.pin_memory,
                drop_last = self.config.hyperparameters.drop_last
                )
        return test_loader
