import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt


def fetch_stock_data(ticker, period='10y', interval='1d'):
    stock_data = yf.download(ticker, period=period, interval=interval)
    return stock_data


def calculate_moving_average(data, day):
    data['20_day_MA'] = data['Close'].shift(1).rolling(window=day).mean()
    data['average_volume'] = data['Volume'].shift(1).mean()
    return data


def apply_trading_strategy(data):
    data['Signal'] = 0  # Default to no position

    buy_condition = ((data['Open'] > 1.1 * data['20_day_MA']) &
                     (data['Volume'] >= data['average_volume']))
    sell_condition = (data['Open'] <= 0.95 * data['20_day_MA'])

    # Loop through data from the first row
    for i in range(1, len(data)):
        if buy_condition.iloc[i]:
            data.at[data.index[i], 'Signal'] = 1
        # If the previous signal is a buy (1) and sell condition is met
        elif sell_condition.iloc[i]:
            data.at[data.index[i], 'Signal'] = -1
        # Otherwise, carry forward the previous signal
        else:
            data.at[data.index[i], 'Signal'] = data.iloc[i-1]['Signal']
    data['Signal'] = data['Signal'].replace(-1, 0)
    data['Positions'] = (data['Signal'].diff())

    return data


def backtest_strategy(data, initial_capital=1000):
    positions = pd.DataFrame(index=stock_data.index).fillna(0.0)
    positions['Stock'] = stock_data['Signal']  # Long positions only

    # Calculate the portfolio value
    portfolio = positions.multiply(stock_data['Close'], axis=0)
    pos_diff = positions.diff()

    # Add 'holdings' to portfolio
    portfolio['holdings'] = ((positions).multiply(
        stock_data['Open'], axis=0)).sum(axis=1)
    # Add 'cash' to portfolio
    portfolio['cash'] = initial_capital - \
        (pos_diff.multiply(stock_data['Open'], axis=0)).sum(axis=1).cumsum()

    # Add 'total' to portfolio
    portfolio['total'] = (portfolio['cash'] + portfolio['holdings'])
    return portfolio


stock_data = fetch_stock_data('SPY')
stock_data = calculate_moving_average(stock_data, 365)
stock_data = apply_trading_strategy(stock_data)
portfolio = backtest_strategy(stock_data, 1000)

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(14, 8))

ax1.plot(stock_data['Close'], label='Close Price')
ax1.plot(stock_data['20_day_MA'], label='20-day Moving Average')
ax1.set_ylabel('Price')
ax1.legend()


ax2.plot(portfolio['total'], label='pos')
ax3.plot(stock_data['Signal'], label='signal')
ax2.set_ylabel('Portfolio Value')
ax2.legend()

plt.show()
