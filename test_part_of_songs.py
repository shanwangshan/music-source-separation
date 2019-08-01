#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 15:32:13 2018

@author: wang9
"""

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
#time domain
data1=np.load('data.npz')
for i in range(10):
    
    te_data=data1['c'].flatten()[i*44100*60:(i+1)*44100*60]
#librosa.output.write_wav('te_data_first.wav', te_data,44100)
    te_lable=data1['d'].flatten()[i*44100*60:(i+1)*44100*60]
#librosa.output.write_wav('te_lable_first.wav', te_lable,44100)
    spec_of_te_data=librosa.stft(te_data,n_fft=1024,hop_length=512)
    spec_of_te_data=spec_of_te_data/np.max(np.abs(spec_of_te_data))
    nb_frames = 40
    test_data=split_in_seqs(spec_of_te_data.T,nb_frames)
#time fre domain

    phase_of_test_data=np.angle(test_data)
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
    estim=np.zeros((2,len(estimate)))
    estim[0,:]=estimate
    estim[1,:]=te_data-estimate
    (sdr, sir, sar, perm)=separation.bss_eval_sources(groundtruth,estim)
    print("sdr = {}, sir = {}, sar = {}".format(sdr, sir, sar))
    librosa.output.write_wav('./test_the_first40/estimate_'+str(i),10*estimate,44100)
