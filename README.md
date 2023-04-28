# ClimateNet

ClimateNet is a Python library for deep learning-based Climate Science. It provides tools for quick detection and tracking of extreme weather events. We also expose models, data sets and metrics to jump-start your research.

## Pre-requisites
- Tested on Linux
- Requires CUDA capable GPU
- Requires anaconda

## Usage

Install the conda environment using 
```sh
conda env create -f conda_env.yml
```

Then, to add conda env kernel to jupyter notebooks, run the following commands: 
```sh
conda activate climatenet
conda install ipykernel
python -m ipykernel install --user --name climatenet --display-name "climatenet"
```
You're good to go to run the files in this repository, minus the data and trained model... which brings us to:
## Downloading and Generating Data
Download the train and test data and the trained model.
To get the data, run the [get_data.py](get_data.py) file.
```bash
python get_data.py
```

Alternatively, you can find the data and a pre-trained model at [https://portal.nersc.gov/project/ClimateNet/](https://portal.nersc.gov/project/ClimateNet/).
To download, you can use the following command:
```sh
wget --recursive --no-parent -R "index.htm*" https://portal.nersc.gov/project/ClimateNet/climatenet_new/
```

If you used the previous command, move the files into the following structure:
```
root
├── data
│   ├── test
│   │   └── data-**.nc <-- These
│   └── train
│       └── data-**.nc <-- These
└── models
    ├── original_baseline_model
    │   ├── config.json
    │   └── weights.pth <-- This
    └── new_baseline_model
        ├── config.json
        └── weights.pth <-- This
```

You can then run [get_data.py](get_data.py) with the `--no-download` flag to only generate the training data.
```bash
python get_data.py --no-download
```

To start multiple runs with different seeds, you can execute the following command:
```bash
python run_multiple.py
```
This will output to a new directory `training_results` with other subdirectories for the types and run number.

You can then generate accompanying plots with the `notebooks/graph_multiple_runs.ipynb` notebook.

The script TestSimple.py is available to test simple models on a subset of the entire dataset. The simple models included are a Random Classifier, Logistic Regression and XGBoost. To modify the model to be run, simply modify the model and model_config arguments in config.py. 

## Rest of original ClimateNet README below

The high-level API makes it easy to train a model from scratch or to use our models to run inference on your own climate data. Just download the model config (or write your own) and train the model using:

```python
config = Config('PATH_TO_CONFIG')
model = CGNet(config)

training_set = ClimateDatasetLabeled('PATH_TO_TRAINING_SET', model.config)
inference_set = ClimateDataset('PATH_TO_INFERENCE_SET', model.config)

model.train(training_set)
model.save_model('PATH_TO_SAVE')

predictions = model.predict(inference_set)
```

You can find an example of how to load our trained model for inference in example.py.

If you are familiar with PyTorch and want a higher degree of control over training procedures, data flow or other aspects of your project, we suggest you use our lower-level modules.
The CGNetModule and Dataset classes conform to what you would expect from standard PyTorch, which means that you can take whatever parts you need and swap out the others for your own building blocks. A quick example of this:

```python
training_data = ... # Plug in your own Dataloader and data handling
cgnet = CGNetModule(classes=3, channels=4)
optimizer = Adam(cgnet.parameters(), ...)      
for features, labels in epoch_loader:
    outputs = softmax(cgnet(features), 1)

    loss = jaccard_loss(outputs, labels) # Or plug in your own loss...
    loss.backward()
    optimizer.step()
    optimizer.zero_grad() 
```

## Data

Climate data can be complex. In order to avoid hard-to-debug issues when reading and interpreting the data, we require the data to adhere to a strict interface when using the high-level abstractions. We're working on conforming to the NetCDF Climate and Forecast Metadata Conventions in order to provide maximal flexibility while still making sure that your data gets interpreted the right way by the models.

## Configurations

When creating a (high-level) model, you need to specify a configuration - on one hand this encourages reproducibility and makes it easy to track and share experiments, on the other hand it helps you avoid issues like running a model that was trained on one variable on an unrelated variable or using the wrong normalisation statistics.
See config.json for an example configuration file.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)

Please cite the relevant papers if this repository is helpful in your research.

Dataset: https://gmd.copernicus.org/articles/14/107/2021/

Methods: https://ai4earthscience.github.io/neurips-2020-workshop/papers/ai4earth_neurips_2020_55.pdf
