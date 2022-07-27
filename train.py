import os
import torch
import torch.nn as nn
import torch.optim as optim
import yacs.config
from config.defaults import get_default_config
from model_training.dataset_loader import DatasetLoader
from model_training.network import Network
from model_training.dataset_creater import DatasetCreater
from tensorboardX import SummaryWriter
from tqdm import tqdm

def load_train_config():
    config = get_default_config()
    config.merge_from_file('config/train.yaml')
    config.freeze()
    return config

def load_val_config():
    config = get_default_config()
    config.merge_from_file('config/test.yaml')
    config.freeze()
    return config

def load_test_config():
    config = get_default_config()
    config.merge_from_file('config/test.yaml')
    config.freeze()
    return config


def get_accuracy(y_prob,y_true):
    y_prob = y_prob >= 0.5
    return (y_true == y_prob).sum().item() / y_true.size(0)

def train(train_config):

    dataset_creater = DatasetCreater(train_config)
    dataset_loader = DatasetLoader(train_config, dataset_creater)
    train_dataloader, val_dataloader = dataset_loader.load_train_val_data()

    writer = SummaryWriter(logdir = 'logs/')
    
    criterion = nn.BCEWithLogitsLoss(reduction = 'mean')
    
    model = Network(train_config)
    
    device = train_config.model.device
    
    model = model.to(device)
    
    if train_config.hyperparameters.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=train_config.hyperparameters.learning_rate,momentum=train_config.hyperparameters.momentum, weight_decay=train_config.hyperparameters.weight_decay)
    
    elif train_config.hyperparameters.optimizer == 'Adam':
        params = [ {'params' : list(model.parameters()),
            'weight_decay':  train_config.hyperparameters.weight_decay }]
        optimizer = torch.optim.Adam(params,lr = train_config.hyperparameters.learning_rate,betas = (train_config.hyperparameters.adam_beta1, train_config.hyperparameters.adam_beta2))
    
    if train_config.hyperparameters.scheduler == 'STEP':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = train_config.hyperparameters.milestones, gamma=train_config.hyperparameters.gamma)
        
    for epoch in range(train_config.hyperparameters.epochs):
        
        model.train()
        train_avg_loss = 0
        train_avg_acc = 0
        for i, (img_rgb,img_nirg,img_ni,labels) in tqdm(enumerate(train_dataloader)):
            optimizer.zero_grad()            
            logits, probs =  model(img_rgb,img_ni)
            loss = criterion(logits, labels)
        
            loss.backward()
            optimizer.step() 

            train_avg_loss += loss.detach().item()
            train_avg_acc += get_accuracy(probs, labels)

        print ("Epoch, Train Loss is", epoch, train_avg_loss/len(train_dataloader))
        print ("Epoch, Train Acc is", epoch, train_avg_acc/len(train_dataloader))

        writer.add_scalar('Train Loss', train_avg_loss/len(train_dataloader) , epoch)
        writer.add_scalar('Train Acc', train_avg_acc/len(train_dataloader) , epoch)

        scheduler.step()

        if (epoch % train_config.hyperparameters.val_freq) == 0:
            
            val_avg_loss = 0
            val_avg_acc = 0
            model.eval()
            for i, (img_rgb,img_nirg,img_ni,labels) in tqdm(enumerate(val_dataloader)):
                with torch.no_grad():
                    logits, probs =  model(img_rgb,img_ni)
                    loss = criterion(logits, labels)
                    val_avg_loss += loss.detach().item()
                    val_avg_acc += get_accuracy(probs,labels)

            print ("Epoch, Val Loss is", epoch, val_avg_loss/len(val_dataloader))  
            print ("Epoch, Val Acc is", epoch, val_avg_acc/len(val_dataloader))

            writer.add_scalar('Val Loss', val_avg_loss/len(val_dataloader) , epoch)
            writer.add_scalar('Val Acc', val_avg_acc/len(val_dataloader) , epoch) 

        if (epoch % train_config.hyperparameters.save_freq) == 0:
            torch.save(model.state_dict(), train_config.model.save_model_dir + str(epoch) + ".pth")

    writer.close()
    return model

if __name__ == "__main__":
    
    train_config = load_train_config()
    model = train(train_config)
    torch.save(model.state_dict(), train_config.model.save_model_dir + "final" + ".pth")

