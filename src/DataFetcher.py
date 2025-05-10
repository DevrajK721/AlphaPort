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
            
            # Change Open Time to Time
            data = data[['Open Time', 'Open', 'Close', 'Volume', 'Low', 'High']]
            data.dropna(inplace=True)
            data.rename(columns={'Open Time': 'Time'}, inplace=True)

            # Compute percentage returns (better for fitting models)
            data['PCT_Returns'] = data['Close'].pct_change()

            data.dropna(inplace=True)
            
            data.to_csv(ticker_file_path, index=False) # Save to CSV 
        
        print("Live Historical Data Fetched and Saved Successfully.")


    def add_indicators(self, data_dir: str = '../data'):
        # Suffixes for the two file sets
        for suffix in ["", "_TESTING"]:
            for ticker in tqdm(
                self.tickers,
                desc=f"Adding Indicators{suffix}",
                unit="file",
                ncols=80,
                bar_format="{desc}: {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                colour="green",
                leave=True,
                dynamic_ncols=True
            ):
                path = os.path.join(data_dir, f"{ticker}{suffix}.csv")

                # 1) Read CSV with Time as datetime index
                df = pd.read_csv(path, parse_dates=["Time"], index_col="Time")

                # 2) EMAs (9, 26)
                df['EMA_9']  = df['Close'].ewm(span=9,  adjust=False).mean()
                df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()

                # 3) MACD (12,26,9)
                ema_fast = df['Close'].ewm(span=12, adjust=False).mean()
                ema_slow = df['Close'].ewm(span=26, adjust=False).mean()
                df['MACD']        = ema_fast - ema_slow
                df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

                # 4) RSI (14)
                delta    = df['Close'].diff()
                gain     = delta.clip(lower=0)
                loss     = -delta.clip(upper=0)
                avg_gain = gain.rolling(14).mean()
                avg_loss = loss.rolling(14).mean()
                rs       = avg_gain / avg_loss
                df['RSI'] = 100 - (100 / (1 + rs))

                # 5) VWAP (intraday reset)
                tp = (df['High'] + df['Low'] + df['Close']) / 3
                vp = tp * df['Volume']
                df['Cum_VP'] = vp.groupby(df.index.date).cumsum()
                df['Cum_V']  = df['Volume'].groupby(df.index.date).cumsum()
                df['VWAP']   = df['Cum_VP'] / df['Cum_V']

                # 6) Bollinger Bands (20,2)
                m20 = df['Close'].rolling(20).mean()
                s20 = df['Close'].rolling(20).std()
                df['BB_Upper'] = m20 + 2 * s20
                df['BB_Lower'] = m20 - 2 * s20

                # 7) ATR (14)
                def rma(x, n): return x.ewm(alpha=1/n, adjust=False).mean()
                hl = df['High'] - df['Low']
                hc = (df['High'] - df['Close'].shift()).abs()
                lc = (df['Low']  - df['Close'].shift()).abs()
                tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
                df['ATR_14'] = rma(tr, 14)

                # 8) ADX (14)
                up = df['High'] - df['High'].shift(1)
                dn = df['Low'].shift(1) - df['Low']
                pos_dm = up.where((up > dn) & (up > 0), 0)
                neg_dm = dn.where((dn > up) & (dn > 0), 0)
                pdi = rma(pos_dm, 14) / rma(tr, 14) * 100
                mdi = rma(neg_dm, 14) / rma(tr, 14) * 100
                dx  = (pdi - mdi).abs() / (pdi + mdi) * 100
                df['ADX']  = rma(dx, 14)
                df['+DI']  = pdi
                df['-DI']  = mdi

                # 9) OBV
                obv = [0]
                vol = df['Volume'].values
                close = df['Close'].values
                for i in range(1, len(df)):
                    if close[i] > close[i-1]:
                        obv.append(obv[-1] + vol[i])
                    elif close[i] < close[i-1]:
                        obv.append(obv[-1] - vol[i])
                    else:
                        obv.append(obv[-1])
                df['OBV'] = obv

                # 10) MFI (14)
                mf = tp * df['Volume']
                pm = mf.where(tp > tp.shift(1), 0).rolling(14).sum()
                nm = mf.where(tp < tp.shift(1), 0).rolling(14).sum()
                mfr = pm / nm
                df['MFI'] = 100 - (100 / (1 + mfr))

                # 11) Stochastic (14,3)
                low14  = df['Low'].rolling(14).min()
                high14 = df['High'].rolling(14).max()
                k      = 100 * (df['Close'] - low14) / (high14 - low14)
                df['%K'] = k
                df['%D'] = k.rolling(3).mean()

                # 12) KAMA (10,2,30)
                n       = 10
                fast_sc = 2/(2+1)
                slow_sc = 2/(30+1)
                chg     = df['Close'].diff(n).abs()
                volsum  = df['Close'].diff().abs().rolling(n).sum()
                er      = chg / volsum
                sc      = (er*(fast_sc-slow_sc)+slow_sc)**2
                kama    = [df['Close'].iat[0]]
                for i in range(1, len(df)):
                    kama.append(kama[-1] + sc.iat[i]*(df['Close'].iat[i]-kama[-1]))
                df['KAMA'] = kama

                # 13) Returns & lags
                df['PCT_Returns'] = df['Close'].pct_change() * 100
                df['Decision']    = np.where(df['PCT_Returns'] > 0.10, 1, -1)
                for i in range(1, 6):
                    df[f'Lag_{i}'] = df['PCT_Returns'].shift(i)

                # 14) Trim warmup
                df = df.iloc[26:]

                # 15) Save back
                df.to_csv(path)

        print("Successfully added indicators to all data files.")
        
    

if __name__ == "__main__":
    dp = DataFetcher()  # One-Liner is all it takes to initialize the DataProcessor class)