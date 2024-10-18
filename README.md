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

## ⚙️ Tech Stack

| Category                      | Technologies                                                                                                                                                                                                                  |
|-------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Programming Languages**      | ![Python](https://img.shields.io/badge/-Python-3776AB?style=flat&logo=python&logoColor=white)                                                                                                                                   |
| **Frameworks**                 | ![PyTorch](https://img.shields.io/badge/-PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)                                                                                                                                |
| **Libraries**                  | ![Falcon](https://img.shields.io/badge/-Falcon-0066CC?style=flat) ![Inflect](https://img.shields.io/badge/-Inflect-FCC624?style=flat) ![Librosa](https://img.shields.io/badge/-Librosa-FFBB00?style=flat)                        |
| **Deep Learning Models**       | ![Transformers](https://img.shields.io/badge/-Transformers-FF9900?style=flat&logo=tensorflow&logoColor=white) ![CBHG](https://img.shields.io/badge/-CBHG-343434?style=flat) ![CNN](https://img.shields.io/badge/-CNN-343434?style=flat)  |
| **Dataset**                    | ![LJ Speech](https://img.shields.io/badge/-LJSpeech-5B618A?style=flat)                                                                                                                                                         |
| **Tools**                      | ![Git](https://img.shields.io/badge/-Git-F05032?style=flat&logo=git&logoColor=white) ![Google Colab](https://img.shields.io/badge/-Google%20Colab-F9AB00?style=flat&logo=googlecolab&logoColor=white) ![Kaggle](https://img.shields.io/badge/-Kaggle-20BEFF?style=flat&logo=kaggle&logoColor=white) |
| **Visualization & Analysis**   | ![Matplotlib](https://img.shields.io/badge/-Matplotlib-11557C?style=flat&logo=python&logoColor=white)                                                                                                                           |

## File Structure

Add here

## Data Extracting 

We used the yFinance Python Library for accessing financial data​.  

Features Extracted -> Open, high, low, close, volume, adj close

## Time Series Forecasting

Time series forecasting is a statistical or machine learning technique that uses historical and current data to predict future values over a period of time or a specific point in the future. It involves building models from historical data and using them to make observations

Forecasting has a range of applications in various industries especially the stock market!

## Time2Vec

Time2Vec is a time encoding mechanism that transforms time-related features into a higher-dimensional space, capturing both linear and periodic patterns. It uses a combination of sine and linear components to effectively represent temporal information. This encoding helps improve the performance of models in time series forecasting tasks by providing a richer representation of time.

## LSTM

Long Short-Term Memory (LSTM) is a type of recurrent neural network (RNN) that can process and analyze sequential data, such as text, speech, and time series. LSTMs are well-suited for problems involving large data sequences and can identify both short-term and long-term dependencies.

### Why LSTM for stock predictions ?
1) Long Short-Term Memory (LSTM) is a type of recurrent neural network (RNN) that can process and analyze sequential data, such as time series. ​

2) Stock prices, are non-stationary and exhibit trends and seasonality. LSTMs can handle these non-linear relationships within the data.​

3) LSTM's can learn long and short term dependencies by selectively retaining information through the memory cell and gates.

4) This characteristic is particularly beneficial in financial time series analysis, where understanding previous market trends is vital for forecasting future price changes. By maintaining a memory cell that stores relevant information over extended periods, LSTMs demonstrate superior performance in capturing subtle nuances and trends within complex trading datasets.

5) Our LSTM model takes 5 features namely, Open, High, Low, Adjusted close and Volume and label is the Closing Price.

## Feature Engineering

Feature engineering is the process of selecting, manipulating and transforming raw data into features that can be used in supervised learning.

It requires using domain knowledge to select and transform the most relevant variables

The goal of feature engineering and selection is to improve the performance of machine learning (ML) algorithms.

### Features Added
1) RSI - Relative Strength Index
The relative strength index (RSI) is a momentum indicator used in technical analysis. RSI measures the speed and magnitude of a security's recent price changes to detect overvalued or undervalued conditions in the price of that security.

On a scale of 0 to 100, RSI reading of 70 or above indicates an overbought condition. A reading of 30 or below indicates an oversold condition.

2) ROC - rate of change
The price rate of change (ROC) is a momentum-based technical indicator that measures the percentage change in price between the current price and the price a certain number of periods ago.

The ROC indicator is plotted against zero, with the indicator moving upwards into positive territory if price changes are to the upside, and moving into negative territory if price changes are to the downside.

It can be used to spot divergences, overbought and oversold conditions, and centerline crossovers.

3) Bollinger bands
Bollinger Bands help gauge the volatility of stocks to determine if they are over or undervalued.

Bands appear on stock charts as three lines that move with the price. The center line is the stock price's 20-day simple moving average (SMA). The upper and lower bands are set at a certain number of standard deviations, usually two, above and below the middle line.

## Transformer Architecture
Transformers are neural network architectures that excel in handling sequential data by using self-attention mechanisms to weigh the importance of each element in the sequence. Unlike traditional models, transformers process input in parallel, allowing for faster training and better handling of long-range dependencies.  


LSTM's might struggle for long range dependencies. They still face challenges when it comes to learning relationships across very distant time steps.  

Unlike traditional recurrent neural networks (RNNs), Transformers leverage attention mechanisms to weigh the significance of each input element concerning others. This mechanism allows them to process information in parallel, enhancing efficiency in analyzing complex financial data patterns. This makes transformers particularly powerful for capturing both short-term and long-term dependencies in a sequence  

## Informer Architecture

informer architecture

