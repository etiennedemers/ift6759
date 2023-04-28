from os import path

from numpy import ndarray

from climatenet.models import CGNet
from climatenet.utils.data import ClimateDatasetLabeled
from climatenet.utils.utils import Config


def finetuning(train_data_path: str, config: Config, save_dir: str = 'results', method: str = 'classifier') -> (
        float, ndarray):
    """
    Finetuning phase of the CGNet model
    :param train_data_dir: The path to the directory containing the dataset (in form of .nc files)
    :param config: The model configuration.
    :param save_dir: The directory to save the results to
    :param method: The method to use for the fine-tuning ('finetune', 'inception', 'classifier')
    :return: Weighted jacard score and ndarray of iou
    """
    config.pretraining = 1
    cgnet = CGNet(config)
    test = ClimateDatasetLabeled(path.join(train_data_dir, 'testSet'), config)
    print('Starting Pretraining Phase')
    cgnet.pretrain(train_data_path)
    print('Saving Model...')
    cgnet.save_model('pretrained_cgnet')
    print('Starting Transition Phase')
    cgnet = CGNet(config, model_path='pretrained_cgnet', save_dir=save_dir, transition_method=method)
    cgnet.train(train_data_path)
    return cgnet.evaluate(test, verbose=True)


def training(train_data_path: str, config: Config, curriculum: bool, save_dir: str = 'results') -> (float, ndarray):
    """
    Training phase of the CGNet model
    :param train_data_path: The path to the directory containing the dataset in subdirectories (in form of .nc files)
    :param config: The model configuration.
    :param curriculum: Whether to use curriculum learning or not (True/False) Curriculum learning is not used by
    default and needs the data to be in a specific
    :param save_dir: The directory to save the results to
    :return:
    """
    config.pretraining = 0
    test = ClimateDatasetLabeled(path.join(train_data_path, 'testSet'), config)
    cgnet = CGNet(config, save_dir=save_dir)
    cgnet.train(train_data_path, curriculum=curriculum)
    return cgnet.evaluate(test, verbose=True)


if __name__ == "__main__":
    config = Config('models_configs/CGNet.json')
    train_data_path = 'data/'
    finetuning(train_data_path, config, save_dir="training_results/finetuning", method='finetune')
    # training(train_data_path,config,False)
