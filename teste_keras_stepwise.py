#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 01:42:40 2017

@author: pedroschoen
"""

import pandas 
import numpy

from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras.layers.core import Dropout
import keras
from imblearn.over_sampling import SMOTE
import teste_ks

seed = 8
numpy.random.seed(seed)


def modelo(train,test,coluna):
    
    tamanho = len(train.columns)     - 1
    
    train=train.values
    test=test.values
    
    X_train = train[:,1:tamanho]
    Y_train = train[:,tamanho]
    X_test =  test[:,1:tamanho]
    Y_test = test[:,tamanho]
    
    sm = SMOTE(kind='regular')
    
    X_resampled, Y_resampled = sm.fit_sample(X_train, Y_train)
    
    layer_1 = 46
    layer_2 = 12 #se zero, comentar a linha do layer_2
    rate = 0.00001
    epoch = 100 #50
    batch = 30
    #decay = 1e-12
    decay = 0.0
    drop = 0.0 #se 0, comentar a linha do dropout
    
    salvar = pandas.read_csv('resultados_stepwise.csv')
    i = len(salvar.index) + 1
    
    # Model Creation, 1 input layer, 1 hidden layer and 1 exit layter
    model = Sequential()
    model.add(Dense(layer_1, input_dim=tamanho-1, init='uniform'))
    model.add(Activation('relu'))
#    model.add(Dropout(drop))
    model.add(Dense(layer_2, init='uniform'))
    # activation='relu'
    model.add(Activation('relu'))
#    model.add(Dropout(drop))
    #model.add(Dense(60, init='uniform'))
    #model.add(Activation('relu'))
    model.add(Dense(1, init='uniform'))
    model.add(Activation('sigmoid'))
    #activation='relu'
    
    opt = keras.optimizers.SGD(lr=rate)
    
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer= opt , metrics=['accuracy'])

    # creating .fit
    model.fit(X_resampled, Y_resampled, nb_epoch=epoch, batch_size=batch,verbose=0)
    
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
    salvar.to_csv("resultados_stepwise.csv",index=False)
    
    return ks_max
    
    
    
base = pandas.read_csv('base_nao_trabalhada.csv')

base['default payment next month'] = (base['default payment next month'] -1)*-1

#dividindo a base entre teste e treino

train1, test1 = train_test_split(base, test_size = 0.2)


columns = ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0',
       'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2',
       'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
       'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']

print ('Modelo com todas as variáveis:')

ks_base = modelo(train1,test1,'Nada') 
ks_max = ks_base
var_min = []
drop = []

stop = True
while stop:
    melhor=0
    for i in columns:
        drop.append(i)
        print ('Testing Removing: ',drop)
        train = train1.drop(i,axis=1)
        test = test1.drop(i,axis=1)
        ks_atual = modelo(train,test,drop)
        
        
        if (ks_atual>ks_max):
            var_min=[]
            var_min.append(i)
            ks_max=ks_atual
        
        if (ks_atual>ks_base):
            melhor=1
            ks_base=ks_atual
        
               
        drop.remove(i)
        
    if (melhor == 0):
        stop=False
    else:
        drop.append(var_min[0])
        columns.remove(var_min[0])
        var_min=[]