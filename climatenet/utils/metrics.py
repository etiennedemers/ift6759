import numpy as np
import torch
import netCDF4
import json
from os import listdir
from tqdm import tqdm
import re
import xarray as xr
import itertools

def get_iou_perClass(confM):
    """
    Takes a confusion matrix confM and returns the IoU per class
    """
    unionPerClass = confM.sum(axis=0) + confM.sum(axis=1) - confM.diagonal()
    iouPerClass = np.zeros(3)
    for i in range(0,3):
        if unionPerClass[i] == 0:
            iouPerClass[i] = 1
        else:
            iouPerClass[i] = confM.diagonal()[i] / unionPerClass[i]
    return iouPerClass
        

def get_cm(pred, gt, n_classes=3):
    cm = np.zeros((n_classes, n_classes))
    for i in range(len(pred)):
        pred_tmp = pred[i].int()
        gt_tmp = gt[i].int()

        for actual in range(n_classes):
            for predicted in range(n_classes):
                is_actual = torch.eq(gt_tmp, actual)
                is_pred = torch.eq(pred_tmp, predicted)
                cm[actual][predicted] += len(torch.nonzero(is_actual & is_pred))  
    return cm


def currScore(output_path, scoring_method='mean'):
    files = listdir('data/train')
    files.extend(listdir('data/test'))
    dates = set(['-'.join(file.split('-')[1:-2]) for file in files])
    score_dict = {}
    for date in tqdm(dates):
        reg = re.compile('data-' + date)
        duplicates = [file for file in files if reg.match(file)]
        if len(duplicates) < 2:
            date_score = -1
        else:
            score_list = []
            for f1, f2 in itertools.combinations(duplicates,2):

                if f1 in listdir('data/test'):
                    base_path = 'data/test/'
                else:
                    base_path = 'data/train/'

                labels1 = torch.tensor(xr.load_dataset(base_path + f1)['LABELS'].values)
                labels2 = torch.tensor(xr.load_dataset(base_path + f2)['LABELS'].values)
                cm = get_cm(labels1, labels2)

                if scoring_method == 'mean':
                    score = get_iou_perClass(cm).mean()
                else:
                    score = get_iou_perClass(cm)

                score_list.append(score)

            date_score = np.array(score_list).mean(axis=0)
            if scoring_method == 'class':
                date_score = list(date_score)
        score_dict.update({date:date_score})

    with open(output_path, 'w') as f:
        json.dump(score_dict,f)
    return