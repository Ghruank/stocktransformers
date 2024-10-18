# Stock Transformers

A Project by Khush Agrawal, Yashvi Mehta and Ghruank Kothare and mentored by Tvisha Vedant and Kindipsingh Malhi

## Aim

In the modern capital market, the price of a stock is often considered to be highly volatile and unpredictable because of various social, financial, political and other dynamic factors that can bring catastrophic financial loss to the investors. This project aims to predict stock prices using transformer architecture by utilising the concept of time series forecasting.

The transformer model has been widely leveraged for natural language processing and computer vision tasks,but is less frequently used for tasks like stock prices prediction. The introduction of time2vec encoding to represent the time series features has made it possible to employ the transformer model for the stock price prediction. We aim to leverage these two effective techniques to discover forecasting ability on volatile stock markets.

## Overview

In this project :

1) We have classified stocks into 2 categories, volatile and non-volatile.
2) For each of them we worked on 3 models namely , LSTM, transformer and informer.
3) We perfomed feature engineering and time2vec to create our datasets.
4) Our models predict the closing price of a stock on a day based on data of previous 100 days.
5) Finally, we concluded it by comparing results across all models and determining which one had the best accuracy.