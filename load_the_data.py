#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 22:46:49 2018

@author: wangshanshan
"""
import librosa 
import numpy as np
import os
#import sounddevice as sd
#import playsound as sd
store_training_data_path=[]
store_testing_data_path=[]
store_training_lable_path=[]
store_testing_lable_path=[]

# External HD path(For Nahid. Don't delete this line.)

ex_hd_path = "/Volumes/My Passport"

# set your current path

current_dir = '.'

root_of_training_data = current_dir + "/DSD100/Mixtures/Dev"
root_of_testing_data = current_dir + "/DSD100/Mixtures/Test"
root_of_training_lable = current_dir + "/DSD100/Sources/Dev"
root_of_testing_lable = current_dir + "/DSD100/Sources/Test"

for path, subdirs, files in sorted(os.walk(root_of_training_data)):
   store_training_data_path.append(path)
for path, subdirs, files in sorted(os.walk(root_of_testing_data)):
   store_testing_data_path.append(path)
for path, subdirs, files in sorted(os.walk(root_of_training_lable)):
   store_training_lable_path.append(path)
for path, subdirs, files in sorted(os.walk(root_of_testing_lable)):
   store_testing_lable_path.append(path)

store_training_data_path=store_training_data_path[1:]
store_testing_data_path=store_testing_data_path[1:]
store_training_lable_path=store_training_lable_path[1:]
store_testing_lable_path=store_testing_lable_path[1:]

tr_data = []
te_data = []
tr_lable=[]
te_lable=[]

for i in store_training_data_path:
    new_data,sr =librosa.core.load(i+'/mixture.wav',sr=None,mono=True, offset=20.0, duration=60, res_type='kaiser_best')
    tr_data.append(new_data)
for i in store_testing_data_path:
    new_data,sr =librosa.core.load(i+'/mixture.wav',sr=None,mono=True, offset=20.0, duration=60, res_type='kaiser_best')
    te_data.append(new_data)
for i in store_training_lable_path:
    new_data,sr=librosa.core.load(i+'/vocals.wav',sr=None,mono=True, offset=20.0, duration=60, res_type='kaiser_best')
    tr_lable.append(new_data)
for i in store_testing_lable_path:
    new_data,sr=librosa.core.load(i+'/vocals.wav',sr=None,mono=True, offset=20.0, duration=60, res_type='kaiser_best')
    te_lable.append(new_data)

#generate the data
tr_data = np.array(tr_data)
tr_data=tr_data/np.max(np.abs(tr_data))
te_data = np.array(te_data)
te_data=te_data/np.max(np.abs(te_data))
tr_lable=np.array(tr_lable)
tr_lable=tr_lable/np.max(np.abs(tr_lable))
te_lable=np.array(te_lable)
te_lable=te_lable/np.max(np.abs(te_lable))
# =============================================================================
# tr_data=np.reshape(tr_data,(1,tr_data.shape[0]*tr_data.shape[1]))
# te_data=np.reshape(te_data,(1,te_data.shape[0]*te_data.shape[1]))
# tr_lable=np.reshape(tr_lable,(1,tr_lable.shape[0]*tr_lable.shape[1]))
# te_lable=np.reshape(te_lable,(1,te_lable.shape[0]*te_lable.shape[1]))
# =============================================================================
output_name='data.npz'
np.savez(output_name,a=tr_data, b=tr_lable, c=te_data, d=te_lable)
