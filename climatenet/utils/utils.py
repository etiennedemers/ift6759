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
    with open(path.join(save_dir,'trainResultsUnion_b1_s42.json'),'r') as f:
        resultUnion_dict = json.load(f)
    train_losses_union = resultUnion_dict['train_losses']
    train_ious_union = resultUnion_dict['train_ious']
    val_ious_union = resultUnion_dict['valid_ious']
    val_losses_union = resultUnion_dict['valid_losses']

    with open(path.join(save_dir,'trainResultsCurr_b1_s42.json'),'r') as f:
        resultCurr_b1_dict = json.load(f)
    train_losses_curr = resultCurr_b1_dict['train_losses']
    train_ious_curr = resultCurr_b1_dict['train_ious']
    val_ious_curr = resultCurr_b1_dict['valid_ious']
    val_losses_curr = resultCurr_b1_dict['valid_losses']

    with open(path.join(save_dir,'trainResultsCurr_b2_s42.json'),'r') as f:
        resultCurr_b2_dict = json.load(f)
    train_losses_curr2 = resultCurr_b2_dict['train_losses']
    train_ious_curr2 = resultCurr_b2_dict['train_ious']
    val_ious_curr2 = resultCurr_b2_dict['valid_ious']
    val_losses_curr2 = resultCurr_b2_dict['valid_losses']

    with open(path.join(save_dir,'trainResultsCurr_b3_s42.json'),'r') as f:
        resultCurr_b3_dict = json.load(f)
    train_losses_curr3 = resultCurr_b3_dict['train_losses']
    train_ious_curr3 = resultCurr_b3_dict['train_ious']
    val_ious_curr3 = resultCurr_b3_dict['valid_ious']
    val_losses_curr3 = resultCurr_b3_dict['valid_losses']

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(10,10))
    plt.subplots_adjust(hspace=0.3)
    l1, = ax1.plot(train_losses_union, label='Normal Training')
    l2, = ax1.plot(train_losses_curr, label='Curriculum Training w/ batch size 1')
    l3, = ax1.plot(train_losses_curr2, label='Curriculum Training w/ batch size 2')
    l4, = ax1.plot(train_losses_curr3, label='Curriculum Training w/ batch size 3')
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Jaccard Loss')

    ax2.plot(train_ious_union, label='Normal Training')
    ax2.plot(train_ious_curr, label='Curriculum Training w/ batch size 1')
    ax2.plot(train_ious_curr2, label='Curriculum Training w/ batch size 2')
    ax2.plot(train_ious_curr3, label='Curriculum Training w/ batch size 3')
    ax2.set_title('Training Mean IOUs')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Mean IOU')

    ax3.plot(val_losses_union, label='Normal Training')
    ax3.plot(val_losses_curr, label='Curriculum Training w/ batch size 1')
    ax3.plot(val_losses_curr2, label='Curriculum Training w/ batch size 2')
    ax3.plot(val_losses_curr3, label='Curriculum Training w/ batch size 3')
    ax3.set_title('Validation Loss')
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('Jaccard Loss')

    ax4.plot(val_ious_union, label='Normal Training')
    ax4.plot(val_ious_curr, label='Curriculum Training w/ batch size 1')
    ax4.plot(val_ious_curr2, label='Curriculum Training w/ batch size 2')
    ax4.plot(val_ious_curr3, label='Curriculum Training w/ batch size 3')
    ax4.set_title('Validation Mean IOUs')
    ax4.set_xlabel('Epochs')
    ax4.set_ylabel('Mean IOU')

    fig.legend([l1,l2,l3,l4],['Normal Training', 'Curriculum Training w/ Batch size 1','Curriculum Training w/ Batch size 2','Curriculum Training w/ Batch size 3'], ncol=2, loc='upper center', bbox_to_anchor=(0.5, 0.05),
          fancybox=True, facecolor='white')
    plt.savefig(path.join(save_dir,'resultsFigures.png'), bbox_inches='tight')

def print_currScore(currScore_path):
    with open(path.join(currScore_path),'r') as f:
        currScore_dict = json.load(f)
    print(np.mean(([currScore_dict[k]  for k in currScore_dict.keys()]),axis=0))
    plt.hist(currScore_dict.values(),bins=20, range=(0,1))
    plt.show()
        
results_plots('./results/Runs')