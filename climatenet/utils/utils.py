import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from os import path
import os
import seaborn as sns

class Config():
    '''
    Abstracts over a model configuration.
    While it currently does not offer any advantages over working with a simple dict,
    it makes it possible to simply add functionality concerning configurations:
    - Checking the validity of a configuration file
    - Automatically loading and saving configuration files

    Parameters
    ----------
    path : str
        The path to the json with the config

    Attributes
    ----------
    architecture : str
        Stores the model architecture type. Currently ignored (only have CGNet), but can be used in the future
    lr : dict
        The learning rate used to train the model
    fields : [str]
        A dictionary mapping from variable names to normalisation statistics
    description : str
        Stores an uninterpreted description string for the model. Put anything you want here.
    '''

    def __init__(self, path: str):
        self.config_dict = json.load(open(path))

        # TODO: Check structure

        self.architecture = self.config_dict['architecture']
        self.lr = self.config_dict['lr']
        self.seed = self.config_dict['seed']
        self.train_batch_size = self.config_dict['train_batch_size']
        self.pred_batch_size = self.config_dict['pred_batch_size']
        self.epochs = self.config_dict['epochs']
        self.fields = self.config_dict['fields']
        self.labels = self.config_dict['labels']
        self.description = self.config_dict['description']

        # Make reproducible
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

    def save(self, save_path: str):
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(self.config_dict, f, ensure_ascii=False, indent=4)

def print_currScore(currScore_path):
    with open(path.join(currScore_path),'r') as f:
        currScore_dict = json.load(f)
    print(np.mean(([currScore_dict[k]  for k in currScore_dict.keys()]),axis=0))
    plt.hist(currScore_dict.values(),bins=20, range=(0,1))
    plt.show()

def results_plots(save_dir: str, file_list, label_list):
    sns.set()
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(10,10))
    plt.subplots_adjust(hspace=0.3)
    axes = [ax1,ax2,ax3,ax4]
    legend_list = [0] * 4
    i=0

    for file,label in zip(file_list,label_list):
        with open(path.join(save_dir,file),'r') as f:
            result_dict = json.load(f)
        train_losses = result_dict['train_losses']
        train_ious = result_dict['train_ious']
        val_ious = result_dict['valid_ious']
        val_losses = result_dict['valid_losses']

        l, = ax1.plot(train_losses, label=label)
        ax2.plot(train_ious, label=label)
        ax3.plot(val_losses, label=label)
        ax4.plot(val_ious, label=label)
        legend_list[i] = l
        i+=1

    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Jaccard Loss')

    ax2.set_title('Training Mean IOUs')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Mean IOU')

    ax3.set_title('Validation Loss')
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('Jaccard Loss')

    ax4.set_title('Validation Mean IOUs')
    ax4.set_xlabel('Epochs')
    ax4.set_ylabel('Mean IOU')

    fig.legend(legend_list,label_list, ncol=2, loc='upper center', bbox_to_anchor=(0.5, 0.05),
          fancybox=True, facecolor='white')
    plt.savefig(path.join(save_dir,'resultsFigures.png'), bbox_inches='tight')

def pretraining_plot(save_dir):
    sns.set()
    with open(path.join(save_dir,'trainResults_pretrain.json'),'r') as f:
        result_dict = json.load(f)
    train_losses = result_dict['train_losses']
    plt.plot(train_losses)
    plt.title('Training MSE')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.savefig(path.join(save_dir,'resultsFigures.png'), bbox_inches='tight')

file_list = ['trainResultsUnion_b3_s442.json','trainResultsCurr_b3_s442.json','trainResultsCurr_b2_s442.json','trainResultsCurr_b1_s442.json']
label_list = ['Normal Training', 'Curriculum Training w/ batch size 3','Curriculum Training w/ batch size 2','Curriculum Training w/ batch size 1']
results_plots('results/OODRuns/',file_list,label_list)