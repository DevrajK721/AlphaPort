# Core Imports 
import numpy as np
import matplotlib.pyplot as plt

# Additional Imports
from src.DataFetcher import DataFetcher as DF
from src.TechnicalFeatures import TechnicalFeatures as TF

# Fetch data using secrets.json file 
data_fetcher = DF(secrets_path='secrets/secrets.json')

# Add technical features to the data (in dataframe)
tech_features = TF(symbols=data_fetcher.tickers, data_dir='data/')
tech_features.populate_dfs()
tech_features.add_technical_features()

