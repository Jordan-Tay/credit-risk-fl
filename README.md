# Credit Risk Forecasting

This project aims to assess the performance of federated learning (FL) in credit risk assessment. The project is broken down to two parts:
1. Corporate S&P credit rating classification (AAA to D)
2. Individual credit default risk (probability)

## Corporate Credit Rating

The dataset is stored in `dataset/`, with an exploratory data analysis performed (EDA) in the notebook `eda.ipynb`. This includes performing feature normalisation, distribution, ANOVA F-test and correlation tests, for feature selection. The experiments are then conducted in `main.ipynb`. This includes:
- n fold cross validation on central MLP/LSTM
- FL simulations on different numbers of workers and data distribution
- Visualisations of loss convergence with different hyperparameters

For the FL experiments, all MLP model checkpoints are stored in `model_checkpoints/` and LSTM in `lstm_model_checkpoints/`.

## Individual Credit Default Risk

Due to the size of the original dataset, it is not uploaded to the repository. The EDA is performed in `eda2.ipynb`, where the data is preprocessing is detailed. The required dataframes are then saved into `.pt` files in `dataset2/`. The same experiments in `main.ipynb` are then performed on the second dataset in `main2.ipynb`. For the FL experiments, all MLP model checkpoints are stored in `model_checkpoints2/` and LSTM in `lstm_model_checkpoints2/`.

## Other files/folders
- `resample.py` contains helper functions for different resampling functions and visualisations for the first dataset
- `fl_simu.py` contains the dataset partitioning strategies and the federated averaging algorithm for each model and dataset
- `net_archs.py` contains the neural network architectures used in both datasets
- `clients/fl_client` contains the abstract class `FLClient` for the FL clients
- `clients/*`, the remaining clients are subclasses of `FLClient` that detail the train and test method for used in each network architecture

## Creating a virtual environment
Create a virtual environment to install the dependencies used in the project:
```
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
python3 -m ipykernel install --user --name=venv
```

Next, select the `venv` kernel in the jupyter notebook, and proceed to run `main.ipynb` and `main2.ipynb` to run the experiments.
