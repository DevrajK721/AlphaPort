# Data Fetching Class 

# Core Imports 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import os 

# Additional Imports 
from binance.client import Client as BC 
import json 
from tqdm import tqdm
from datetime import datetime, timedelta
from typing import * 

class DataFetcher:
    def __init__(self, secrets_path: str = '../secrets/secrets.json'):
        # Verify existence of secrets.json file 
        if not os.path.exists(secrets_path):
            raise FileNotFoundError(f"secrets.json file not found at {secrets_path}. Please ensure the provided path is correct (Hint: Use `pwd` in command line to verify current directory)")
        else:
            self.secrets_path = secrets_path
            print(f"secrets.json file found at {secrets_path}. Beginning initialization of DataProcessor class.")
        
        # Load secrets from the JSON file
        with open(self.secrets_path, 'r') as file:
            vals = json.load(file)
            self.binance_api_key = vals['BINANCE_API_KEY']
            self.frequency = vals['Trading Frequency (Yearly/Monthly/Weekly/Daily/Hourly/Minutely)']
            self.binance_api_secret = vals['BINANCE_API_SECRET']
            self.end_date = vals['Ending Date (YYYY-MM-DD)']
            self.start_date = vals['Starting Date (YYYY-MM-DD)']
            self.base_currency = vals['Base Currency']
            self.n = 50 # Fetch at least 50 data points, for worst case. 
            self.tickers = vals['Tickers of Interest']

        # Check whether all information has been loaded correctly 
        if not all([self.binance_api_key, self.binance_api_secret, self.start_date, self.end_date, self.base_currency, self.tickers]):
            raise ValueError("One or more required fields are missing in the secrets.json file.")
        
        # Initialize the Binance Client
        self.binance_client = BC(self.binance_api_key, self.binance_api_secret)
        print("Binance client initialized successfully.")

        print("All required fields loaded successfully from secrets.json.")
        print(f"{len(self.tickers)} tickers loaded successfully.")
        print(f"Frequency: {self.frequency}")
        print(f"Starting date: {self.start_date}")
        print(f"Ending date: {self.end_date}")
        print(f"Base currency: {self.base_currency}")
        print(f"Tickers: {self.tickers}")
        print("Initialization of DataProcessor class completed successfully.")

        if self.frequency == 'Daily':
            interval = BC.KLINE_INTERVAL_1DAY
        elif self.frequency == "Minutely":
            interval = BC.KLINE_INTERVAL_1MINUTE
        elif self.frequency == 'Hourly':
            interval = BC.KLINE_INTERVAL_1HOUR
        elif self.frequency == 'Weekly':
            interval = BC.KLINE_INTERVAL_1WEEK
        elif self.frequency == 'Monthly':
            interval = BC.KLINE_INTERVAL_1MONTH
        elif self.frequency == 'Yearly':
            interval = BC.KLINE_INTERVAL_1YEAR
        else:
            raise ValueError("Invalid frequency. Choose from 'Daily', 'Weekly', 'Monthly', or 'Yearly' and update in secrets.json file.")

        # Convert tickers to trading pairs 
        self.tickers = [f"{ticker}{self.base_currency}" for ticker in self.tickers]

        # Find the path to the project root 
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(project_root, 'data')
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
        self.volatilities = {}

        for ticker in tqdm(self.tickers, desc="Fetching Crypto Data", unit="pair",
            ncols=80, bar_format="{desc}: {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            colour="red", leave=True, dynamic_ncols=True):
            # Check if the ticker data already exists
            ticker_file_path = os.path.join(data_dir, f"{ticker}.csv")
            if os.path.exists(ticker_file_path):
                print(f"Data for {ticker} already exists at {ticker_file_path}. Skipping download.")
                # Load existing data to calculate volatility
                data = pd.read_csv(ticker_file_path)
                returns = data['Close'].pct_change().dropna()
                self.volatilities[ticker] = returns.std()
                continue 

            symbol = ticker 
            columns = ['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close Time', 
            'Quote Asset Volume', 'Number of Trades', 'Taker Buy Base Asset Volume', 
            'Taker Buy Quote Asset Volume', 'Ignore']
            data = pd.DataFrame(self.binance_client.get_historical_klines(symbol, interval, self.start_date, self.end_date), columns=columns)
            data['Open Time'] = pd.to_datetime(data['Open Time'], unit='ms')
            data['Close'] = data['Close'].astype(float)
            data['Open'] = data['Open'].astype(float)
            data['High'] = data['High'].astype(float)
            data['Low'] = data['Low'].astype(float)
            data['Volume'] = data['Volume'].astype(float)
            
            # Keep Open Time and Close ONLY 
            data = data[['Open Time', 'Open', 'Close', 'Volume', 'Low', 'High']]
            data.dropna(inplace=True)
            data.rename(columns={'Open Time': 'Time'}, inplace=True)
            data.to_csv(ticker_file_path, index=False) # Save to CSV 
            # Calculate volatility based on the trading frequency without scaling
            # Just use the standard deviation of returns for the native frequency
            returns = data['Close'].pct_change().dropna()
            self.volatilities[ticker] = returns.std()
            

        print("All Historical Data Fetched and Saved Successfully.")

        for ticker in tqdm(self.tickers, desc="Fetching Test Crypto Data", unit="pair",
            ncols=80, bar_format="{desc}: {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            colour="blue", leave=True, dynamic_ncols=True):
            # Check if the ticker data already exists
            ticker_file_path = os.path.join(data_dir, f"{ticker}_TESTING.csv")
            if os.path.exists(ticker_file_path):
                print(f"Testing Data for {ticker} already exists at {ticker_file_path}. Skipping download.")
                continue 
            
            symbol = ticker 
            columns = ['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close Time', 
            'Quote Asset Volume', 'Number of Trades', 'Taker Buy Base Asset Volume', 
            'Taker Buy Quote Asset Volume', 'Ignore']
            self.end_date = (datetime.now() - timedelta(days=0)).strftime("%Y-%m-%d")
            self.start_date = (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d")
            data = pd.DataFrame(self.binance_client.get_historical_klines(symbol, interval, self.start_date, self.end_date), columns=columns)
            data['Open Time'] = pd.to_datetime(data['Open Time'], unit='ms')
            data['Close'] = data['Close'].astype(float)
            data['Open'] = data['Open'].astype(float)
            data['High'] = data['High'].astype(float)
            data['Low'] = data['Low'].astype(float)
            data['Volume'] = data['Volume'].astype(float)
            
            # Keep Open Time and Close ONLY 
            data = data[['Open Time', 'Open', 'Close', 'Volume', 'Low', 'High']]
            data.dropna(inplace=True)
            data.rename(columns={'Open Time': 'Time'}, inplace=True)
            data.to_csv(ticker_file_path, index=False) # Save to CSV 
        
        print("Live Historical Data Fetched and Saved Successfully.")


    

if __name__ == "__main__":
    dp = DataFetcher()  # One-Liner is all it takes to initialize the DataProcessor class)