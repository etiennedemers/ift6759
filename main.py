from os import path

from numpy import ndarray

from climatenet.models import CGNet
from climatenet.utils.data import ClimateDatasetLabeled
from climatenet.utils.utils import Config


def finetuning(train_data_dir: str, config: Config, save_dir: str = 'results', method: str = 'classifier'):
    config.pretraining = 1
    cgnet = CGNet(config)
    test = ClimateDatasetLabeled(path.join(train_data_path, 'testSet'), config)
    print('Starting Pretraining Phase')
    cgnet.pretrain()
    print('Saving Model...')
    cgnet.save_model('pretrained_cgnet')
    print('Starting Transition Phase')
    cgnet = CGNet(config, model_path='pretrained_cgnet', save_dir=save_dir, transition_method=method)
    cgnet.train(train_data_dir)
    cgnet.evaluate(test, verbose=True)


def training(train_data_path: str, config: Config, curriculum: bool, save_dir: str = 'results') -> (float, ndarray):
    config.pretraining = 0
    test = ClimateDatasetLabeled(path.join(train_data_path, 'testSet'), config)
    cgnet = CGNet(config, save_dir=save_dir)
    cgnet.train(train_data_path, curriculum=curriculum)
    return cgnet.evaluate(test, verbose=True)


if __name__ == "__main__":
    config = Config('models_configs/CGNet.json')
    train_data_path = 'data/ood/'
    finetuning(train_data_path, config, method='finetune')
    # training(train_data_path,config,False)
