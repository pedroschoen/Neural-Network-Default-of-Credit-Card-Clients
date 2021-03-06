#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 01:42:40 2017

@author: pedroschoen
"""
seed = 8

import pandas 
import numpy
numpy.random.seed(seed)

import tensorflow as tf
tf.set_random_seed(seed)

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras.layers.core import Dropout
import keras
from imblearn.over_sampling import SMOTE
import teste_ks



def modelo(train,test,coluna):
    
    
    

    
    X_train = train[columns]
    Y_train = train['default payment next month']
    X_test =  test[columns]
    Y_test = test['default payment next month']

    
    tamanho = len(X_train.columns) + 1
    
    X_train= X_train.values
    Y_train= Y_train.values
    X_test= X_test.values
    Y_test= Y_test.values
    
    sm = SMOTE(kind='regular')
    
    X_resampled, Y_resampled = sm.fit_sample(X_train, Y_train)
    
    layer_1 = 46
    layer_2 = 12 #se zero, comentar a linha do layer_2
    rate = 0.01
    epoch = 200 #50
    batch = 128
    decay = 1e-12
    #decay = 0.0
    drop = 0.5 #se 0, comentar a linha do dropout
    
    salvar = pandas.read_csv('resultados.csv')
    i = len(salvar.index) + 1
    
    # Model Creation, 1 input layer, 1 hidden layer and 1 exit layter
    model = Sequential()
    model.add(Dense(layer_1, input_dim=tamanho-1, init='uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(drop))
    model.add(Dense(layer_2, init='uniform'))
    #activation='relu'
    model.add(Activation('relu'))
    model.add(Dropout(drop))
    #model.add(Dense(60, init='uniform'))
    #model.add(Activation('relu'))
    model.add(Dense(1, init='uniform'))
    model.add(Activation('sigmoid'))
    #activation='relu'
    
    opt = keras.optimizers.SGD(lr=rate,decay=decay)
    
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer= opt , metrics=['accuracy'])

    # creating .fit
    model.fit(X_resampled, Y_resampled, nb_epoch=epoch, batch_size=batch) #,verbose=0
    
    scores = model.evaluate(X_test, Y_test)
    print ()
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    
    # predicting model
    predictions = model.predict(X_test)
    
    predictions=pandas.DataFrame(predictions,columns=['SCORE'])
    s=pandas.DataFrame(Y_test,columns=['CLASSE'])
    x=predictions.join(s)
    
    ks_max = teste_ks.KS(x)
    
    salvar.loc[i] = [ks_max,layer_1,layer_2,rate,epoch,batch,decay,drop,coluna]
    salvar.to_csv("resultados.csv",index=False)
    
    return ks_max
    
    
    
base_treino = pandas.read_csv('base_treino.csv',sep=";")


base_treino['default payment next month'] = (base_treino['default payment next month'] -1)*-1

base_teste = pandas.read_csv('base_teste.csv',sep=";")

base_teste['default payment next month'] = (base_teste['default payment next month'] -1)*-1

columns = ['LIM_BAL_1', 'LIM_BAL_2', 'LIM_BAL_3',
       'SEX_1', 'SEX_2', 'EDUCATION_1', 'EDUCATION_2', 'EDUCATION_3',
       'EDUCATION_4', 'EDUCATION_5', 'EDUCATION_6', 'EDUCATION_7',
       'MARRIAGE_1', 'MARRIAGE_2', 'MARRIAGE_3', 'MARRIAGE_4', 'AGE_1',
       'AGE_2', 'AGE_3', 'PAY_0_1', 'PAY_0_2', 'PAY_0_3', 'PAY_2_1', 'PAY_2_2',
       'PAY_2_3', 'PAY_3_1', 'PAY_3_2', 'PAY_3_3', 'PAY_4_1', 'PAY_4_2',
       'PAY_4_3', 'PAY_5_1', 'PAY_5_2', 'PAY_5_3', 'PAY_6_1', 'PAY_6_2',
       'PAY_6_3', 'BILL_AMT1_1', 'BILL_AMT1_2', 'BILL_AMT1_3', 'BILL_AMT2_1',
       'BILL_AMT2_2', 'BILL_AMT2_3', 'BILL_AMT3_1', 'BILL_AMT3_2',
       'BILL_AMT3_3', 'BILL_AMT4_1', 'BILL_AMT4_2', 'BILL_AMT4_3',
       'BILL_AMT5_1', 'BILL_AMT5_2', 'BILL_AMT5_3', 'BILL_AMT6_1',
       'BILL_AMT6_2', 'BILL_AMT6_3', 'PAY_AMT1_1', 'PAY_AMT1_2', 'PAY_AMT1_3',
       'PAY_AMT2_1', 'PAY_AMT2_2', 'PAY_AMT2_3', 'PAY_AMT3_1', 'PAY_AMT3_2',
       'PAY_AMT3_3', 'PAY_AMT4_1', 'PAY_AMT4_2', 'PAY_AMT4_3', 'PAY_AMT5_1',
       'PAY_AMT5_2', 'PAY_AMT5_3', 'PAY_AMT6_1', 'PAY_AMT6_2', 'PAY_AMT6_3']



ks_base = modelo(base_treino,base_teste,'Nada') 

