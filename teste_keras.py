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
    

seed = 8
numpy.random.seed(seed)


base = pandas.read_csv('base_nao_trabalhada.csv')

#dividindo a base entre teste e treino

train, test = train_test_split(base, test_size = 0.2)




#dividir base entre teste e treino antiga
#
#len(base)
#
#msk = numpy.random.rand(len(base)) < 0.8
#train = base[msk]
#test = base[~msk]
#
#len(train)
#len(test)


train=train.values
test=test.values

X_train = train[:,1:23]
Y_train = train[:,24]
X_test =  test[:,1:23]
Y_test = test[:,24]

sm = SMOTE(kind='regular')

X_resampled, Y_resampled = sm.fit_sample(X_train, Y_train)

layer_1 = 15
layer_2 = 15
rate = 0.00001
epoch = 300
batch = 30
#decay = 1e-12
decay = 0.0

# Model Creation, 1 input layer, 1 hidden layer and 1 exit layter
model = Sequential()
model.add(Dense(15, input_dim=22, init='uniform'))
model.add(Activation('relu'))
#model.add(Dropout(0.1))
model.add(Dense(15, init='uniform'))
# activation='relu'
model.add(Activation('relu'))
##model.add(Dropout(0.1))
#model.add(Dense(60, init='uniform'))
#model.add(Activation('relu'))
model.add(Dense(1, init='uniform'))
model.add(Activation('sigmoid'))
#activation='relu'

opt = keras.optimizers.SGD(lr=0.00001)

# Compile model
model.compile(loss='binary_crossentropy', optimizer= opt , metrics=['accuracy'])
#loss=binary_crossentropy
#optimizer='adam'


# creating .fit
model.fit(X_resampled, Y_resampled, nb_epoch=300, batch_size=30)




# evaluate the model
scores = model.evaluate(X_test, Y_test)
print ()
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))



# predicting model
predictions = model.predict(X_test)

predictions=pandas.DataFrame(predictions,columns=['SCORE'])
s=pandas.DataFrame(Y_test,columns=['CLASSE'])
x=predictions.join(s)
#numpy.savetxt("testeks.csv", x, fmt='%.2f',delimiter=";",)

x['mau']= 1 - x.CLASSE


x['bucket'] = pandas.qcut(x.SCORE, 10)

grouped = x.groupby('bucket', as_index = False)

#numpy.savetxt("testeks.csv", x, fmt='%.2f',delimiter=";",)


agg1 = grouped.min().SCORE
 
agg1 = pandas.DataFrame(grouped.min().SCORE, columns = ['min_scr'])
 
agg1['max_scr'] = grouped.max().SCORE
 
agg1['bads'] = grouped.sum().mau

agg1['goods'] = grouped.sum().CLASSE
 
agg1['total'] = agg1.bads + agg1.goods

 
agg2 = (agg1.sort_values(by = 'min_scr')).reset_index(drop = True)
 
agg2['odds'] = (agg2.goods / agg2.bads).apply('{0:.2f}'.format)
 
agg2['bad_rate'] = (agg2.bads / agg2.total).apply('{0:.2%}'.format)
 
 
agg2['ks'] = numpy.round(((agg2.bads / x.mau.sum()).cumsum() - (agg2.goods / x.CLASSE.sum()).cumsum()), 4) * 100
  
flag = lambda x: '<----' if x == agg2.ks.max() else ''
 
agg2['max_ks'] = agg2.ks.apply(flag)
   
print ()
print (agg2)
#
#numpy.savetxt("teste_scores", predictions, delimiter=";", fmt='%s')
#numpy.savetxt("teste_y", Y_test, delimiter=";", fmt='%s')
#
