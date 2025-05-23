# Add technical features to the dataset
import pandas as pd 
import numpy as np
from sklearn.preprocessing import StandardScaler 
from typing import *

class TechnicalFeatures:
    def __init__(self, symbols=List[str], data_dir:str = 'data/'):
        self.symbols = symbols
        self.data_dir = data_dir 
        self.dfs = {}
        self.testing_dfs = {}

    def populate_dfs(self):
        for sym in self.symbols:
            df = pd.read_csv(f"{self.data_dir}{sym}.csv", parse_dates=["Time"], index_col="Time")
            df_testing = pd.read_csv(f"{self.data_dir}{sym}_testing.csv", parse_dates=["Time"], index_col="Time")
            df = df.sort_index()
            df_testing = df_testing.sort_index()
            self.dfs[sym] = df
            self.testing_dfs[sym] = df_testing

    def add_technical_features(self, commission_tol:float=0.001):
        for sym in self.symbols:
            df = self.dfs[sym]
            df_testing = self.testing_dfs[sym]
            
            # Log Return 
            df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
            df_testing['Log_Return'] = np.log(df_testing['Close'] / df_testing['Close'].shift(1))

            # Moving Averages
            df["SMA_12"] = df["Close"].rolling(window=12).mean()
            df_testing["SMA_12"] = df_testing["Close"].rolling(window=12).mean()
            df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
            df_testing["EMA_12"] = df_testing["Close"].ewm(span=12, adjust=False).mean()

            # Relative Strength Index (RSI)
            delta = df["Close"].diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            df["RSI_14"] = 100 - (100 / (1 + rs))

            delta_testing = df_testing["Close"].diff()
            gain_testing = delta_testing.clip(lower=0)
            loss_testing = -delta_testing.clip(upper=0)
            avg_gain_testing = gain_testing.rolling(window=14).mean()
            avg_loss_testing = loss_testing.rolling(window=14).mean()
            rs_testing = avg_gain_testing / avg_loss_testing
            df_testing["RSI_14"] = 100 - (100 / (1 + rs_testing))

            # Rolling Volatility
            df["Volatility_12"] = df["Log_Return"].rolling(window=12).std()
            df_testing["Volatility_12"] = df_testing["Log_Return"].rolling(window=12).std()

            # ATR Proxy 
            df["ATR_12"] = (df["High"] - df["Low"]).rolling(window=12).mean()
            df_testing["ATR_12"] = (df_testing["High"] - df_testing["Low"]).rolling(window=12).mean()

            # On-Balance Volume (OBV)
            direction = np.where(df["Close"] > df["Close"].shift(1), 1, 
                                 np.where(df["Close"] < df["Close"].shift(1), -1, 0))
            df["OBV"] = (direction * df["Volume"]).cumsum()

            direction_testing = np.where(df_testing["Close"] > df_testing["Close"].shift(1), 1,
                                         np.where(df_testing["Close"] < df_testing["Close"].shift(1), -1, 0))
            df_testing["OBV"] = (direction_testing * df_testing["Volume"]).cumsum()

            # MACD Histogram
            ema_short = df["Close"].ewm(span=12, adjust=False).mean()
            ema_long  = df["Close"].ewm(span=26, adjust=False).mean()
            macd_line = ema_short - ema_long
            signal    = macd_line.ewm(span=9, adjust=False).mean()
            df["MACD_HIST"] = macd_line - signal

            ema_short_testing = df_testing["Close"].ewm(span=12, adjust=False).mean()
            ema_long_testing  = df_testing["Close"].ewm(span=26, adjust=False).mean()
            macd_line_testing = ema_short_testing - ema_long_testing
            signal_testing    = macd_line_testing.ewm(span=9, adjust=False).mean()
            df_testing["MACD_HIST"] = macd_line_testing - signal_testing

            # Bollinger Bands
            bb_mid    = df["Close"].rolling(20).mean()
            bb_std    = df["Close"].rolling(20).std()
            df["bb_upper"] = bb_mid + 2 * bb_std
            df["bb_lower"] = bb_mid - 2 * bb_std
           
            bb_mid_testing    = df_testing["Close"].rolling(20).mean()
            bb_std_testing    = df_testing["Close"].rolling(20).std()
            df_testing["bb_upper"] = bb_mid_testing + 2 * bb_std_testing
            df_testing["bb_lower"] = bb_mid_testing - 2 * bb_std_testing

            # Normalize the features
            feature_cols = [
            "Log_Return", "SMA_12", "EMA_12", "RSI_14",
            "Volatility_12", "ATR_12", "OBV",
            "MACD_HIST", "bb_upper", "bb_lower"
            ]

            scaler = StandardScaler()
            df[feature_cols] = scaler.fit_transform(df[feature_cols])
            df_testing[feature_cols] = scaler.transform(df_testing[feature_cols])

            # Create a simple Binary Target Variable
            df["Forward_Return"] = df["Log_Return"].shift(-1)
            df["Label"] = np.where(df["Forward_Return"] > commission_tol, 1, 0)
            df_testing["Forward_Return"] = df_testing["Log_Return"].shift(-1)
            df_testing["Label"] = np.where(df_testing["Forward_Return"] > commission_tol, 1, 0)
        
            # Drop NaN values
            df.dropna(subset=["Label"])
            df_testing.dropna(subset=["Label"])
            df.dropna(inplace=True)
            df_testing.dropna(inplace=True)

            # Save the modified DataFrames back to dictionary
            self.dfs[sym] = df
            self.testing_dfs[sym] = df_testing

        print(f"Technical features added for all symbols.")

            

            
            


