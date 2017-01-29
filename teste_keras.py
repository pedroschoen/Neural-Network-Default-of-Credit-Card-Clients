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


# Model Creation, 1 input layer, 1 hidden layer and 1 exit layter
model = Sequential()
model.add(Dense(40, input_dim=22, init='uniform', activation='relu'))
model.add(Dense(4, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))
#activation='relu'

opt = keras.optimizers.SGD(lr=0.00001)

# Compile model
model.compile(loss='binary_crossentropy', optimizer=opt , metrics=['accuracy'])
#loss=binary_crossentropy
#optimizer='adam'


# creating .fit
model.fit(X_train, Y_train, nb_epoch=30, batch_size=5)




# evaluate the model
scores = model.evaluate(X_test, Y_test)
print ()
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))



# predicting model
predictions = model.predict(X_test)

numpy.savetxt("teste_scores", predictions, delimiter=";", fmt='%s')
numpy.savetxt("teste_y", Y_test, delimiter=";", fmt='%s')

