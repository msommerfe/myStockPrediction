'''
Created on 26 Jul 2020

@author: marco
'''
from pandas_datareader import data
from pandas_datareader._utils import RemoteDataError
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
from datetime import datetime
from numpy.ma.core import getdata
from cProfile import label
from matplotlib.pyplot import title

START_DATE = '2005-01-01'
END_DATE = str(datetime.now().strftime('%Y-%m-%d'))

UK_STOCK = 'UU.L'
USA_STOCK = 'AMZN'
DAX = '^GDAXI'

def get_stats(stock_data):
    return {
        'last':np.mean(stock_data.tail(1)),
        'short_mean':np.mean(stock_data.tail(20)),
        'long_mean':np.mean(stock_data.tail(200)),
        'short_rolling': stock_data.rolling(window=20).mean(),
        'long_rolling': stock_data.rolling(window=200).mean()
    }
    
def clean_data(stock_data, col):
    weekdays = pd.date_range(start=START_DATE, end=END_DATE)
    clean_data = stock_data[col].reindex(weekdays)
    return clean_data.fillna(method='ffill')

def create_plot(stock_data, ticker):
    stats = get_stats(stock_data)
    plt.style.use('dark_background')
    plt.subplots(figsize=(12,8))
    plt.plot(stock_data, label=ticker)
    plt.plot(stats['short_rolling'], label='20 day rolling mean')
    plt.plot(stats['long_rolling'], label='200 day rolling mean')
    plt.xlabel('Date')
    plt.ylabel('Adj Close')
    plt.legend()
    plt.title('Stock price over Time')
    plt.show()

def get_data(ticker):

    try:
        stock_data = data.DataReader(ticker,
                                    'yahoo',
                                    START_DATE,
                                    END_DATE)

        stock_data = stock_data.apply (pd.to_numeric, errors='coerce')
        stock_data = stock_data.dropna()
        #print(stock_data)
        #print(stock_data[0:5])
        X=[]
        Y=[]
        df=[]
        df = stock_data[['Adj Close']]
        df.insert(0,'High', stock_data[['High']])
        datesetSize = stock_data.shape[0]
        
        for i in range(0,10):
            randomI = random.randint(50, datesetSize)
            dfSubset = df[(randomI-50) : randomI]           #Achtung das bedeutet der Index randomI ist NICHT mehr teil von dfSubset 
            tomorrow = df['Adj Close'][randomI]             #Hier wird aber genau auf den Index randomI zugegriffen
            today = df['Adj Close'][randomI-1]
            if tomorrow > today:
                Y.append(1)
            else:
                Y.append(0)
         
            X.append(np.array(dfSubset))

        adj_close = clean_data(stock_data, 'Adj Close')
        #create_plot(adj_close, ticker)
   

        
        
    except RemoteDataError:
        print('No data found for {t}'.format(t=ticker))

get_data(DAX)