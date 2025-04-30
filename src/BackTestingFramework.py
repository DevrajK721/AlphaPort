import backtrader as bt
import math
import numpy as np


# Fix CSV column mapping to match file: Time,Open,Close,Volume,Low,High
class GenericCSVData(bt.feeds.GenericCSVData):
    params = (
        ("nullvalue", float("NaN")),
        ("dtformat", "%Y-%m-%d %H:%M:%S"),
        ("tmformat", "%H:%M:%S"),
        ("datetime", 0),
        ("open", 1),
        ("high", 5),  # column index 5 is High
        ("low", 4),  # column index 4 is Low
        ("close", 2),  # column index 2 is Close
        ("volume", 3),  # column index 3 is Volume
        ("openinterest", -1),
        ("timeframe", bt.TimeFrame.Minutes),
        ("compression", 1),
        ("headers", True),
    )


# Strategy Definition - Test Case for now until we make list of all indicators we are interested in
"""
Example Strategy
Disclaimer: Not realistic, just for testing purposes
Buy when:
- EMA(9) > EMA(21)
- RSI(14) > 50
- MACD Histogram > 0

Sell when:
- Any of the above conditions are not met anymore 
"""


class TestStrategy(bt.Strategy):
    def __init__(self):
        # 2) Bind the close-price series so you can index it
        self.dataclose = self.datas[0].close

        self.ema9 = bt.indicators.EMA(self.dataclose, period=9)
        self.ema21 = bt.indicators.EMA(self.dataclose, period=21)
        self.rsi = bt.indicators.RSI(self.dataclose, period=14)
        self.macd = bt.indicators.MACD(
            self.dataclose, period_me1=12, period_me2=26, period_signal=9
        )
        self.order = None

    def notify_order(self, order):
        if order.status in [order.Completed, order.Canceled, order.Margin]:
            self.order = None  # clear past order

    def next(self):
        if self.order:
            return

        price = self.dataclose[0]
        cash = self.broker.getcash()
        size = cash / price

        buy_signal = (
            self.ema9[0] > self.ema21[0]
            and self.rsi[0] > 50
            and (self.macd.macd[0] - self.macd.signal[0]) > 0
            and (self.macd.macd[0] < 0) 
            and (self.macd.signal[0] < 0)
        )

        if not self.position and buy_signal:
            self.order = self.buy(size=size)
            print(f"BUY  @ {self.data.datetime.date(0)} price={price:.6f} size={size}")

        # sell when any condition breaks
        sell_signal = (
            self.ema9[0] < self.ema21[0]
            or self.rsi[0] < 50
            or (self.macd.macd[0] - self.macd.signal[0]) < 0
        )

        if self.position and sell_signal:
            self.order = self.sell(size=self.position.size)
            print(
                f"SELL @ {self.data.datetime.date(0)} price={price:.6f} size={self.position.size}"
            )


def run_backtest():
    cerebro = bt.Cerebro()
    cerebro.addstrategy(TestStrategy)
    cerebro.broker.setcash(600.0)

    # Load minute data but compress into 1-hour bars
    data = GenericCSVData(
        dataname="../data/DOTUSDT_TESTING.csv",
        timeframe=bt.TimeFrame.Minutes,
        compression=60,    # 60 × 1-minute = 1-hour bars
    )
    cerebro.adddata(data)

    # Hourly Sharpe
    cerebro.addanalyzer(
        bt.analyzers.SharpeRatio,
        timeframe=bt.TimeFrame.Minutes,
        compression=60,
        riskfreerate=0.0,
        _name="sharpe_hourly",
    )
    # Daily returns for average daily return calculation
    cerebro.addanalyzer(
        bt.analyzers.TimeReturn,
        timeframe=bt.TimeFrame.Days,
        compression=1,
        _name="dailyreturns",
    )
    # Drawdown
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")

    results = cerebro.run()
    strat   = results[0]

    final_value = cerebro.broker.getvalue()

    # Hourly Sharpe
    sharpe_h = strat.analyzers.sharpe_hourly.get_analysis().get("sharperatio", None)

    # Daily returns analyzer → dictionary: {date: return}
    dr = strat.analyzers.dailyreturns.get_analysis()
    arr_daily = np.array(list(dr.values()))
    if arr_daily.size:
        avg_daily_ret = arr_daily.mean() * 100
    else:
        avg_daily_ret = None

    # Max Drawdown
    draw = strat.analyzers.drawdown.get_analysis()["max"]["drawdown"]

    print(f"{'='*20} DATA BACKTEST RESULTS {'='*20}")
    print(f"Final Portfolio Value: {final_value:.2f}")
    if sharpe_h is not None:
        print(f"Sharpe Ratio (hourly): {sharpe_h:.4f}")
    else:
        print("Sharpe Ratio (hourly): N/A")
    if avg_daily_ret is not None:
        print(f"Average Daily Return: {avg_daily_ret:.2f}%")
    else:
        print("Average Daily Return: N/A")
    print(f"Max Drawdown: {draw:.2f}%")


if __name__ == "__main__":
    run_backtest()
