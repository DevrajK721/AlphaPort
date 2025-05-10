# Basic backtrader implementation for when strategy is ready to be tested (SKELETON CODE)

# Imports 
import numpy as np 
import backtrader as bt
from backtrader.feeds import GenericCSVData
from datetime import datetime
import pandas as pd 
import matplotlib.pyplot as plt
import glob
import os 

# Define custom CSV data class
class MyCSV(bt.feeds.GenericCSVData):
    """Map your CSV header to Backtrader.  Only *extra* columns go in `lines`."""
    lines  = ('ema9', 'ema26')                       # everything else is built-in
    params = dict(
        dtformat   ='%Y-%m-%d %H:%M:%S',             # Time column format
        nullvalue  =float('NaN'),                   
        datetime   =0,  open=1, high=5, low=4, close=2, volume=3, openinterest=-1,
        ema9       =6,  ema26=7                     # map two EMA columns
    )   

# Define the strategy (THIS IS THE SKELETON CODE SECTION)
class EMACross(bt.Strategy):
    params = dict( stake_perc=0.95 )                 

    def __init__(self):
        self.cross   = bt.ind.CrossOver(self.data.ema9, self.data.ema26)  # +1 / −1
        self.order   = None
        self.portval = []                         

    def notify_order(self, order):
        if order.status == order.Completed:
            act = 'BUY' if order.isbuy() else 'SELL'
            print(f'{self.data.datetime.datetime(0)}  {act}'
                  f' @{order.executed.price:.5f}  size={order.executed.size:.0f}')
        self.order = None                           

    # ------- main loop --------------------------------------------------------
    def next(self):
        self.portval.append((self.datetime.datetime(0), self.broker.getvalue()))
        if self.order:                                                
            return

        cash, close = self.broker.getcash(), self.data.close[0]

        if not self.position and self.cross > 0:                      
            size = int(self.p.stake_perc * cash / close)
            if size:
                self.order = self.buy(size=size)

        elif self.position and self.cross < 0:                        
            self.order = self.close()

    def stop(self):
        if self.position:
            self.close()

# Run Backtest and gather metrics + plot
def run_backtest(csv_path, start_cash=600.0):
    cerebro = bt.Cerebro(stdstats=False)
    cerebro.broker.setcash(start_cash)
    cerebro.broker.setcommission(commission=0.001)
    cerebro.broker.set_coc(True)
    cerebro.addstrategy(EMACross)

    data = MyCSV(
        dataname   = csv_path,
        timeframe  = bt.TimeFrame.Minutes,
        compression= 1
    )
    cerebro.adddata(data)

    cerebro.addanalyzer(bt.analyzers.TimeReturn, timeframe=bt.TimeFrame.Days, _name='daily')
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, timeframe=bt.TimeFrame.Days,
                        riskfreerate=0.0, annualize=True, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='dd')

    strat = cerebro.run()[0]

    daily = strat.analyzers.daily.get_analysis()
    avg_d = np.mean(list(daily.values())) * 100 if daily else 0.0

    sh    = strat.analyzers.sharpe.get_analysis().get('sharperatio')
    sh_txt= f'{sh:.2f}' if sh is not None else 'N/A'

    dd_max= strat.analyzers.dd.get_analysis()['max']['drawdown'] * 100

    print('\n=== Performance ===')
    print(f'Average Daily Return:  {avg_d:.2f}%')
    print(f'Annualized Sharpe:     {sh_txt}')
    print(f'Max Drawdown:          {dd_max:.2f}%')

    # Plot results 
    dates, equity = zip(*strat.portval)          # unpack the tuples
    plt.figure(figsize=(12,6))
    plt.plot(dates, equity, linewidth=1.2)
    plt.title('Equity Curve')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

