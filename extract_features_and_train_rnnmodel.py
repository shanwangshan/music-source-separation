#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 15:52:04 2018

@author: wangshanshan
"""

import librosa 
import numpy as np
import os
from keras.models import Sequential
from keras.layers import GRU,LSTM,TimeDistributed
from keras.layers.core import Dense, Activation, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
#import playsound as sd
#import sounddevice as sd
import sklearn.metrics as metrics
from mir_eval import separation
#stack the data in a row by flattening them
data=np.load('data.npz')
tr_data=data['a'].flatten()
tr_lable=data['b'].flatten()
te_data=data['c'].flatten()
te_lable=data['d'].flatten()
#extract the features of training data and training lable and normalize them
spec_of_tr_data=librosa.stft(tr_data,n_fft=1024,hop_length=512)
spec_of_tr_data=spec_of_tr_data/np.max(np.abs(spec_of_tr_data))
spec_of_tr_lable=librosa.stft(tr_lable,n_fft=1024,hop_length=512)
spec_of_tr_lable=spec_of_tr_lable/np.max(np.abs(spec_of_tr_lable))
#used for testing part
spec_of_te_data=librosa.stft(te_data,n_fft=1024,hop_length=512)
spec_of_te_lable=librosa.stft(te_lable,n_fft=1024,hop_length=512)
# the function below is from one of our exercise
def split_in_seqs(data, subdivs):
    """
    Splits a long sequence matrix into sub-sequences.
        Eg: input: data = MxN  sub-sequence length (subdivs) = 2
            output = M/2 x 2 x N
        
    :param data: Array of one or two dimensions 
    :param subdivs: integer value representing a sub-sequence length
    :return: array of dimension = input array dimension + 1 
    """
    if len(data.shape) == 1:
        if data.shape[0] % subdivs:
            data = data[:-(data.shape[0] % subdivs), :]
        data = data.reshape((data.shape[0]//subdivs, subdivs, 1))
    elif len(data.shape) == 2:
        if data.shape[0] % subdivs:
            data = data[:-(data.shape[0] % subdivs), :]
        data = data.reshape((data.shape[0]//subdivs, subdivs, data.shape[1]))
    return data

def get_rnn_model(in_data, out_data):
    _model=Sequential()
    
    print(in_data.shape)
    print(out_data.shape)
    #print(out_data.shape[])
    _model.add(LSTM(units=512,return_sequences=True, input_shape=in_data.shape[1:],dropout=0.0)) 
    _model.add(LSTM(units=512,return_sequences=True, dropout=0.0))
    _model.add(LSTM(units=512,return_sequences=True, dropout=0.0)) 
    _model.add(TimeDistributed(Dense(out_data.shape[-1], init='he_normal', activation='relu')))
   # print(out_data.shape[-1])
    
    _model.summary()
    return _model
#define the sequence length
nb_frames = 40

#split the spectragram into sequences
train_data=split_in_seqs(spec_of_tr_data.T,nb_frames)
train_lable=split_in_seqs(spec_of_tr_lable.T,nb_frames)
test_data=split_in_seqs(spec_of_te_data.T,nb_frames)
test_lable=split_in_seqs(spec_of_te_lable.T,nb_frames)
#save the splitted spectragram of the test data and test lable
output_name='te_data_and_lable.npz'
np.savez(output_name,a=test_data, b=test_lable)

#deep learning starts
model = get_rnn_model(train_data, train_lable)
checkpoint = ModelCheckpoint('./best_weights.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
early_stopping = EarlyStopping(monitor='val_loss',  patience=30, verbose=1, mode='auto')
callbacks_list = [checkpoint, early_stopping]
model.compile(optimizer='Adam', loss='mse')
his = model.fit(np.abs(train_data),np.abs(train_lable), batch_size=128,epochs=200,validation_split=0.3,callbacks=callbacks_list)
loss = his.history['loss']
val_loss = his.history['val_loss']
plt.plot(loss)
plt.hold(True)
plt.plot(val_loss)
plt.savefig('loss-vs-val_loss')
model.save('my_modle.h5')







