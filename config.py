import argparse


def get_config_parser():
    parser = argparse.ArgumentParser(description="Run an experiment")
    
    data = parser.add_argument_group("Data")
    data.add_argument(
        "--train_path", type=str, default='data', help="Train Path"
    )
    data.add_argument(
        "--standardize", type=bool, default=True, help="Train Path"
    )
    
    model = parser.add_argument_group("Model")
    model.add_argument(
        "--model",
        type=str,
        choices=[ "LogReg", "XGBoost", "Random"],
        default="Random",
        help="name of the model to run (default: %(default)s).",
    )
    model.add_argument(
        "--model_config",
        type=str,
        default='./models_configs/Random.json',
        help="Path to model config json file"
    )
    return parser
