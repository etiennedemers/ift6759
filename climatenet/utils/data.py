from torch.utils.data import Dataset
from os import listdir, path, makedirs
from shutil import copy
import json
import re
import xarray as xr
import numpy as np
from climatenet.utils.utils import Config

class ClimateDataset(Dataset):
    '''
    The basic Climate Dataset class. 

    Parameters
    ----------
    path : str
        The path to the directory containing the dataset (in form of .nc files)
    config : Config
        The model configuration. This allows to automatically infer the fields we are interested in 
        and their normalisation statistics

    Attributes
    ----------
    path : str
        Stores the Dataset path
    fields : dict
        Stores a dictionary mapping from variable names to normalisation statistics
    files : [str]
        Stores a sorted list of all the nc files in the Dataset
    length : int
        Stores the amount of nc files in the Dataset
    '''
  
    def __init__(self, path: str, config: Config):
        self.path: str = path
        self.fields: dict = config.fields
        
        self.files: [str] = [f for f in sorted(listdir(self.path)) if f[-3:] == ".nc"]
        self.length: int = len(self.files)
      
    def __len__(self):
        return self.length

    def normalize(self, features: xr.DataArray):
        for variable_name, stats in self.fields.items():   
            var = features.sel(variable=variable_name).values
            var -= stats['mean']
            var /= stats['std']

    def get_features(self, dataset: xr.Dataset):
        features = dataset[list(self.fields)].to_array()
        self.normalize(features)
        return features.transpose('time', 'variable', 'lat', 'lon')

    def __getitem__(self, idx: int):
        file_path: str = path.join(self.path, self.files[idx]) 
        dataset = xr.load_dataset(file_path)
        return self.get_features(dataset)

    @staticmethod
    def collate(batch):
        return xr.concat(batch, dim='time')

class ClimateDatasetLabeled(ClimateDataset):
    '''
    The labeled Climate Dataset class. 
    Corresponds to the normal Climate Dataset, but returns labels as well and batches accordingly
    '''

    def __getitem__(self, idx: int):
        file_path: str = path.join(self.path, self.files[idx]) 
        dataset = xr.load_dataset(file_path)
        return self.get_features(dataset), dataset['LABELS']

    @staticmethod 
    def collate(batch):
        data, labels = map(list, zip(*batch))
        return xr.concat(data, dim='time'), xr.concat(labels, dim='time')
    
    
def save_datefiles(files,output_path):
    for file in files:
        if file in listdir('data/test'):
            base_path = 'data/test/'
        else:
             base_path = 'data/train/'
        makedirs(output_path, exist_ok=True)
        copy(path.join(base_path, file), output_path)

    
def sample_subset(output_path,n,method='rand'):
    files = listdir('data/train')
    files.extend(listdir('data/test'))
    if method == 'rand':
        select_files = np.random.permutation(files)[:n]
    if method == 'curriculum':
        with open('data/currScore.json','r') as f:
            score_dict = json.load(f)
        select_dates = [k for k, v in sorted(score_dict.items(), key=lambda item: item[1], reverse=True)][:n]
        select_files = []
        for date in select_dates:
            reg = re.compile('data-' + date)
            select_files.append([file for file in files if reg.match(file)][0])

    save_datefiles(select_files,output_path)
    

def create_datasets(data_path,ood=False):
    with open('data/currScore.json','r') as f:
            score_dict = json.load(f)

    files = listdir('data/train')
    test_files = listdir('data/test')
    files.extend(test_files)

    if not ood: test_files = []

    test_dates = set(['-'.join(file.split('-')[1:-2]) for file in test_files])
    for k in test_dates:
        score_dict.pop(k, None)

    unique = [k for k, v in score_dict.items() if score_dict[k]==-1]
    val_dates = unique[:40]
    if not ood: test_dates = unique[40:][:61]

    tresh_dict = {'test':'','validation':'','simple':0.55,'medium':0.475,'hard':-10}

    for dataset, tresh in tresh_dict.items():
        print(dataset)
        if dataset == 'validation':
            select_dates = val_dates
        elif dataset == 'test':
            select_dates = test_dates
            print(test_dates)
        else:
            select_dates = [k for k, v in score_dict.items() if score_dict[k]>tresh]
        
        if dataset != 'test':
            for k in select_dates:
                score_dict.pop(k, None)

        select_files = []
        for date in select_dates:
            reg = re.compile('data-' + date)
            select_files.extend([file for file in files if reg.match(file)])

        output_path = path.join(data_path, dataset+'Set')
        save_datefiles(select_files,output_path)
        if dataset in ['simple', 'medium', 'hard']:
            output_path = path.join(data_path, 'unionSet')
            save_datefiles(select_files,output_path)
        
