from climatenet.utils.utils import Config
from main import finetuning, training


def run_multiple(number_of_runs: int = 5):
    """
    Run multiple experiments with different seeds and return the average results
    :param number_of_runs: The number of runs to do
    :return: The average results of the runs, saving generated graphs
    """
    train_data_path = 'data/'
    config = Config('models_configs/CGNet.json')
    results = []
    for i in range(number_of_runs):
        print(f'Run {i + 1}/{number_of_runs}')
        results.append(training(train_data_path, config, False, save_dir=f'training_results/run_{i}'))


if __name__ == '__main__':
    run_multiple()
