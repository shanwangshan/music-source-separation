#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 11:48:11 2018

@author: wangshanshan
"""
import librosa 
import numpy as np
from keras.models import load_model
from mir_eval import separation 
#import sounddevice as sd

#load the time domain data
data1=np.load('data.npz')
te_data=data1['c'].flatten()
te_lable=data1['d'].flatten()
#load the time-fre domain data
data=np.load('te_data_and_lable.npz')
test_data=data['a']
test_data=test_data/np.max(np.abs(test_data))
phase_of_test_data=np.angle(test_data)
#test_lable=data['d']
#phase_of_test_data=np.angle(test_lable)

model=load_model('my_modle.h5')
model.load_weights('./best_weights.hdf5')
estimated_magnitude=model.predict(np.abs(test_data))
#estimated_magnitude= np.abs(test_lable)


#phase=np.angle(test_data)
pre=estimated_magnitude*np.exp(1j*phase_of_test_data)
estimate=np.reshape(pre,(pre.shape[0]*pre.shape[1], pre.shape[2]))
#estimate1=np.reshape(pre,(pre.shape[2],pre.shape[0]*pre.shape[1]))

estimate=librosa.istft(estimate.T,hop_length=512)
#sd.play(estimate)
te_lable=te_lable[:len(estimate)]
te_data=te_data[:len(estimate)]
groundtruth=np.zeros((2,len(estimate)))
groundtruth[0,:]=te_lable
groundtruth[1,:]=te_data-te_lable
estim = np.zeros((2,len(estimate)))
estim[0,:] = estimate
estim[1,:] = te_data - estimate
(sdr, sir, sar, perm)=separation.bss_eval_sources(groundtruth,estim)
print("sdr={},sar={}".format(sdr,sar))
librosa.output.write_wav('estimate_test_data.wav', estimate,44100)
