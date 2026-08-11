import datetime
from dateutil.relativedelta import relativedelta
from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
from scipy import stats


def downloadData(ticker):
    start = datetime.datetime.now() - relativedelta(years=5)
    end = datetime.datetime.now()

    data = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False
    )

    # Handle yfinance MultiIndex columns
    if isinstance(data.columns, np.ndarray):
        pass

    if hasattr(data.columns, "levels"):
        if len(data.columns.levels) > 1:
            data.columns = data.columns.get_level_values(0)

    data.index.name = "Date"

    return data


def shortenData(stockData, marketData, year):
    if year == 5:
        return stockData, marketData

    cutoff = datetime.datetime.now() - relativedelta(years=year)

    stockData = stockData[stockData.index > cutoff]
    marketData = marketData[marketData.index > cutoff]

    return stockData, marketData


def processBeta(stockData, marketData, year, frequency, adjustment=0):

    stockData, marketData = shortenData(
        stockData,
        marketData,
        year
    )

    stock = (
        stockData["Adj Close"]
        .resample(frequency)
        .last()
        .pct_change(fill_method=None)
    )

    market = (
        marketData["Adj Close"]
        .resample(frequency)
        .last()
        .pct_change(fill_method=None)
    )

    combined = np.column_stack(
        (market.values, stock.values)
    )

    mask = np.isfinite(combined).all(axis=1)

    market = market.values[mask]
    stock = stock.values[mask]

    if len(market) < 2:
        raise ValueError(
            f"Not enough data for {year} year {frequency} calculation."
        )

    beta = calculateBeta(stock, market)

    regressionData = stats.linregress(
        market,
        stock
    )

    return (
        adjustBeta(beta, adjustment),
        market,
        stock,
        regressionData
    )


def calculateBeta(stockData, marketData):

    covariance = np.cov(
        stockData,
        marketData
    )[0, 1]

    variance = np.var(
        marketData,
        ddof=1
    )

    return covariance / variance


def adjustBeta(beta, adjustment):

    for _ in range(adjustment):
        beta = 0.67 * beta + 0.33

    return beta


def subPlot(title, x, y, colour, regressionData, position):

    plots = {
        1: 321,
        2: 322,
        3: 323,
        4: 324,
        5: 325,
        6: 326
    }

    ax = plt.subplot(plots[position])

    ax.set_title(title)

    ax.scatter(
        x,
        y,
        color=colour,
        alpha=0.6
    )

    slope = regressionData.slope
    intercept = regressionData.intercept

    x_line = np.linspace(
        np.min(x),
        np.max(x),
        100
    )

    y_line = slope * x_line + intercept

    ax.plot(
        x_line,
        y_line,
        color="black"
    )

    ax.axhline(
        0,
        color="gray",
        linewidth=0.5
    )

    ax.axvline(
        0,
        color="gray",
        linewidth=0.5
    )

    ax.set_xlabel("Market Returns")
    ax.set_ylabel("Stock Returns")


def beta(
    tickers,
    market="^GSPC",
    adjusted=0
):

    colours = cycle(
        ["b", "g", "r", "c", "m", "y"]
    )

    marketData = downloadData(market)

    for ticker in tickers:

        stockData = downloadData(ticker)

        beta1m5y, y1, x1, rD1 = processBeta(
            stockData,
            marketData,
            5,
            "ME",
            adjusted
        )

        beta1m3y, y2, x2, rD2 = processBeta(
            stockData,
            marketData,
            3,
            "ME",
            adjusted
        )

        beta1w5y, y3, x3, rD3 = processBeta(
            stockData,
            marketData,
            5,
            "W",
            adjusted
        )

        beta1w3y, y4, x4, rD4 = processBeta(
            stockData,
            marketData,
            3,
            "W",
            adjusted
        )

        beta1w1y, y5, x5, rD5 = processBeta(
            stockData,
            marketData,
            1,
            "W",
            adjusted
        )

        beta1d1y, y6, x6, rD6 = processBeta(
            stockData,
            marketData,
            1,
            "D",
            adjusted
        )

        plt.figure(figsize=(12.5, 10))

        colour = next(colours)

        subPlot(
            "Monthly 5 Years",
            x1,
            y1,
            colour,
            rD1,
            1
        )

        subPlot(
            "Monthly 3 Years",
            x2,
            y2,
            colour,
            rD2,
            2
        )

        subPlot(
            "Weekly 5 Years",
            x3,
            y3,
            colour,
            rD3,
            3
        )

        subPlot(
            "Weekly 3 Years",
            x4,
            y4,
            colour,
            rD4,
            4
        )

        subPlot(
            "Weekly 1 Year",
            x5,
            y5,
            colour,
            rD5,
            5
        )

        subPlot(
            "Daily 1 Year",
            x6,
            y6,
            colour,
            rD6,
            6
        )

        plt.suptitle(
            f"{ticker} Beta",
            fontsize=18
        )

        plt.tight_layout()

        plt.show()

        print(
            f"""
{ticker} Betas:

Monthly 5 Years : {beta1m5y:.4f}
Monthly 3 Years : {beta1m3y:.4f}
Weekly 5 Years  : {beta1w5y:.4f}
Weekly 3 Years  : {beta1w3y:.4f}
Weekly 1 Year   : {beta1w1y:.4f}
Daily 1 Year    : {beta1d1y:.4f}
"""
        )


# Run the calculator
beta(["AAPL"])