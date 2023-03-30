from climatenet.models import CGNet
from climatenet.utils.utils import Config
from climatenet.utils.data import ClimateDatasetLabeled
from os import path

def finetuning(train_data_dir,config,save_dir: str = 'results',method: str = 'classifier'):
    config.pretraining = 1
    cgnet = CGNet(config,transition_method=method)
    print('Starting Pretraining Phase')
    cgnet.pretrain()
    print('Saving Model...')
    cgnet.save_model('pretrained_cgnet')
    print('Starting Finetuning Phase')
    cgnet = CGNet(model_path='pretrained_cgnet',save_dir=save_dir)
    cgnet.train(train_data_dir)

def training(train_data_path,config,curriculum,save_dir: str = 'results'):
    config.pretraining = 0
    test = ClimateDatasetLabeled(path.join(train_data_path, 'testSet'), config)
    cgnet = CGNet(config,save_dir=save_dir)
    cgnet.train(train_data_path,curriculum=curriculum)
    cgnet.evaluate(test,verbose=True)


if __name__ == "__main__":
    config = Config('models_configs/CGNet.json')
    train_data_path = 'data/ood/'
    finetuning(train_data_path,config)