from climatenet.utils.utils import Config
from main import finetuning, training
from random import randint


def run_multiple_union(number_of_runs: int = 5, random_seed: bool = True):
    """
    Run multiple experiments with different seeds and return the average results
    :param number_of_runs: The number of runs to do
    :param random_seed: Use a random seed or not.
    :return: The average results of the runs, saving generated graphs
    """
    train_data_path = 'data/'
    for i in range(number_of_runs):
        seed = None
        if random_seed:
            seed = randint(0, 100000)
        config = Config('models_configs/CGNet.json', random_seed=seed)
        print(f'Run {i + 1}/{number_of_runs}')
        training(train_data_path, config, curriculum=False, save_dir=f'training_results/union/run_{i}')


def run_multiple_curriculum(number_of_runs: int = 5, random_seed: bool = True):
    """
    Run multiple experiments with different seeds and return the average results
    :param number_of_runs: The number of runs to do
    :param random_seed: Use a random seed or not.
    :return: The average results of the runs, saving generated graphs
    """
    train_data_path = 'data/'
    for i in range(number_of_runs):
        seed = None
        if random_seed:
            seed = randint(0, 100000)
        config = Config('models_configs/CGNet.json', random_seed=seed)
        print(f'Run {i + 1}/{number_of_runs}')
        training(train_data_path, config, curriculum=True, save_dir=f'training_results/curriculum/run_{i}')


def run_multiple_pretraining(number_of_runs: int = 5, random_seed: bool = True,
                             methods: list = ['finetune', 'inception', 'classifier']):
    """
    Run multiple experiments with different seeds and return the average results
    :param number_of_runs: The number of runs to do
    :param random_seed: Use a random seed or not.
    :param methods: The types of pretraining to use, default is all three (finetune, inception, classifier)
    :return: The average results of the runs, saving generated graphs
    """
    train_data_path = 'data/'
    for i in range(number_of_runs):
        seed = None
        if random_seed:
            seed = randint(0, 100000)
        config = Config('models_configs/CGNet.json', random_seed=seed)
        print(f'Run {i + 1}/{number_of_runs}')
        for method in methods:
            print(f'Pretraining method: {method}')
            finetuning(train_data_path, config, save_dir=f'training_results/pretraining/{type}/run_{i}', method=method)


if __name__ == '__main__':
    run_multiple_union(number_of_runs=5, random_seed=True)
    run_multiple_curriculum(number_of_runs=5, random_seed=True)
    run_multiple_pretraining(number_of_runs=5, random_seed=True)
