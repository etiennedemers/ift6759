import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from os import path
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
    
def results_plots(save_dir: str):
    sns.set()
    with open(path.join(save_dir,'trainResults.json'),'r') as f:
        result_dict = json.load(f)
    train_losses = result_dict['train_losses']
    train_ious = result_dict['train_ious']

    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,5))
    ax1.plot(train_losses)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Jaccard Loss')

    ax2.plot(train_ious)
    ax2.set_title('Training Mean IOUs')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Mean IOU')
    plt.savefig(path.join(save_dir,'resultsFigures.png'), bbox_inches='tight')

results_plots('/Users/vincentlongpre/Documents/Devoirs/IFT6759/ift6759/results/') 
        
