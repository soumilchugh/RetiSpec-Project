import os
import torch
import torch.nn as nn
import yacs.config
from model_training.dataset_loader import DatasetLoader
from model_training.network import Network
from model_training.dataset_creater import DatasetCreater
from config.defaults import get_default_config
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import seaborn as sns
from tqdm import tqdm

def compute_confusion_matrix(pred, labels):
    cm = confusion_matrix(labels,pred)
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, fmt='g', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation

    # labels, title and ticks
    ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
    ax.set_title('Confusion Matrix'); 
    ax.xaxis.set_ticklabels(['Forest', 'River']); ax.yaxis.set_ticklabels(['Forest', 'River'])
    fig = ax.get_figure()
    fig.savefig("confusion_matrix.png")

def load_test_config():
    config = get_default_config()
    config.merge_from_file('config/test.yaml')
    config.freeze()
    return config

def get_accuracy(y_prob,y_true):
    y_prob = y_prob >= 0.5
    return (y_true == y_prob).sum().item() / y_true.size(0)

def test(test_config):
    
    dataset_creater = DatasetCreater(test_config)
    dataset_loader = DatasetLoader(test_config, dataset_creater)
    test_dataloader = dataset_loader.load_test_data()
    model = Network(test_config) 
    device = test_config.model.device
    
    model = model.to(device)
    model.load_state_dict(torch.load(test_config.model.model_test_dir, map_location=test_config.model.device)) 
    model.eval()
    test_avg_acc = 0
    labels_list = []
    prob_list = []
    for i, (img_rgb,img_nirg,img_ni,labels) in tqdm(enumerate(test_dataloader)):
        with torch.no_grad():
            img_rgb = img_rgb.to(device)
            logits, probs =  model(img_rgb,img_ni)
            test_avg_acc += get_accuracy(probs,labels)
            probs = probs.cpu().numpy()[0][0]
            labels = labels.cpu().numpy()[0][0]
            labels_list.append(labels)
            prob_list.append(probs)

    print ("Test Acc is", test_avg_acc/len(test_dataloader))

    return labels_list, prob_list


if __name__ == "__main__":    
    test_config = load_test_config()
    labels_list, prob_list = test(test_config)
    y_pred = np.where(np.array(prob_list) > 0.5, 1, 0)

    compute_confusion_matrix(y_pred, labels_list)
    print(classification_report(labels_list, y_pred))

