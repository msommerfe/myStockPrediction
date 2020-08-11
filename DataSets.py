'''
Created on 26 Jul 2020

@author: marco
'''
from pandas_datareader import data
from pandas_datareader._utils import RemoteDataError
import tensorflow as tf
import pandas as pd
import numpy as np
import random
from datetime import datetime

START_DATE = '2005-01-01'
END_DATE = str(datetime.now().strftime('%Y-%m-%d'))

UK_STOCK = 'UU.L'
USA_STOCK = 'AMZN'
DAX = '^GDAXI'
ticker = DAX
 

def get_stock_data(numberDataSet, rollingDays, isRandom):

    try:
        stock_data = data.DataReader(ticker,
                                    'yahoo',
                                    START_DATE,
                                    END_DATE)

        stock_data = stock_data.apply (pd.to_numeric, errors='coerce')
        stock_data = stock_data.dropna()
        X=[]
        Y=[]
        df=[]
        df = stock_data[['Adj Close']]
        #df.insert(0,'High', stock_data[['High']])
        datesetSize = stock_data.shape[0]
        
        for i in range(0,numberDataSet):
            if isRandom:
                index = random.randint(rollingDays, datesetSize-1)
            else:
                index = datesetSize-1 - i

            dfSubset = df[(index-rollingDays) : index]    #Achtung das bedeutet der Index randomI ist NICHT mehr teil von dfSubset 
            tomorrow = df['Adj Close'][index]             #Hier wird aber genau auf den Index randomI zugegriffen
            today = df['Adj Close'][index-1]
            if tomorrow > today:
                Y.append(1)
            else:
                Y.append(0)
         
            X.append(np.array(dfSubset))

        X = np.array(X).reshape(-1, X[0].shape[0], X[0].shape[1])
        #X = np.array(X).reshape(-1, X[0].shape[0], X[0].shape[1],1) Falls ein X[i] Datensatz mehrere Spalten hat

        X = tf.keras.utils.normalize(X, axis=1)
        Y = np.array(Y)
        
        return (X,Y)

    except RemoteDataError:
        print('No data found for {t}'.format(t=ticker))

#model.save('myFristModel')
#nmodel = tf.keras.models.load_model('myFristModel')
#prediction = nmodel.predict([x_test])
#print(prediction[1])

def get_csv_data(numberDataSet, rollingDays, isRandom):

    df = pd.read_csv("D:/dev/Datasets/sin_fkt.csv", delimiter=';',decimal=",")
    df = df[['Sinus']]
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.dropna()
    X=[]
    Y=[]
    datesetSize = df.shape[0]
    
    for i in range(0,numberDataSet):
        if isRandom:
            index = random.randint(rollingDays, datesetSize-1)
        else:
            index = datesetSize-1 - i
        dfSubset = df[(index-rollingDays) : index]          #Achtung das bedeutet der Index randomI ist NICHT mehr teil von dfSubset 
        tomorrow = df['Sinus'][index]                       #Hier wird aber genau auf den Index randomI zugegriffen
        today = df['Sinus'][index-1]
        if tomorrow > today:
            Y.append(1)
        else:
            Y.append(0)
        
        X.append(np.array(dfSubset))


    X = np.array(X).reshape(-1, X[0].shape[0], X[0].shape[1])
    #X = np.array(X).reshape(-1, X[0].shape[0], X[0].shape[1],1) Falls ein X[i] Datensatz mehrere Spalten hat

    X = tf.keras.utils.normalize(X, axis=1)
    Y = np.array(Y)
    
    #np.save('features.npy',X)
    #np.save('labels.npy',Y)
    return (X,Y)




get_csv_data(1800, 10, False)