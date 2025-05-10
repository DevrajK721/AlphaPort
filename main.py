# Core Imports 
import numpy as np
import matplotlib.pyplot as plt

# Additional Imports
import src.DataFetcher as DF 
import src.BackTestingFramework as BTF 

# Fetch data using secrets.json file 
data_fetcher = DF.DataFetcher(secrets_path='secrets/secrets.json')
data_fetcher.add_indicators(data_dir='data')


BTF.run_backtest('data/FIDAUSDT.csv')
