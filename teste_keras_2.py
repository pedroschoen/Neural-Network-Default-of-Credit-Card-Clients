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
import keras


seed = 8        
numpy.random.seed(seed)


base_treino = pandas.read_csv('base_treino.csv',sep=';')
base_teste = pandas.read_csv('teste.csv',sep=';')

#dividindo a base entre teste e treino




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


base_treino = base_treino.values
base_teste = base_teste.values

X_train = base_treino[:,1:23]
Y_train = base_treino[:,24]
X_test =  base_teste[:,1:23]
Y_test = base_teste[:,24]


# Model Creation, 1 input layer, 1 hidden layer and 1 exit layter
model = Sequential()
model.add(Dense(60, input_dim=22, init='uniform', activation='relu'))
model.add(Dense(4, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))
#activation='relu'

opt = keras.optimizers.SGD(lr=0.000001)

# Compile model
model.compile(loss='binary_crossentropy', optimizer=opt , metrics=['accuracy'])
#loss=binary_crossentropy
#optimizer='adam'


# creating .fit
model.fit(X_train, Y_train, nb_epoch=150, batch_size=30)




# evaluate the model
scores = model.evaluate(X_test, Y_test)
print ()
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))



# predicting model
predictions = model.predict(X_test)

numpy.savetxt("teste_scores", predictions, delimiter=";", fmt='%s')
numpy.savetxt("teste_y", Y_test, delimiter=";", fmt='%s')

