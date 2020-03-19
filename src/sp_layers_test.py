import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from sp_layers import SPLayer

def fbank_test():
    conf = {
        "feature_type": "fbank",
        "sample_rate": 16000,
        "num_mel_bins": 40,
        "use_energy": False
        }
    fn = "file:testdata/100-121669-0000.wav"
    pipe = "pipe:flac -c -d -s testdata/103-1240-0005.flac |"
    sample_rate, waveform1 = utils.load_wave(fn)    
    sample_rate, waveform2 = utils.load_wave(pipe)
    waveform1 = torch.from_numpy(waveform1)
    waveform2 = torch.from_numpy(waveform2)
    lengths = [waveform1.shape[0], waveform2.shape[0]]
    max_length = max(lengths)
    padded_waveforms = torch.zeros(2, max_length)
    padded_waveforms[0, :lengths[0]] += waveform1
    padded_waveforms[1, :lengths[1]] += waveform2
    layer = SPLayer(conf)
    
    features, feature_lengths = layer(padded_waveforms, lengths)
    print(features)
    print(feature_lengths)

def specaug_fbank_test():
    conf = {
        "feature_type": "fbank",
        "sample_rate": 16000,
        "num_mel_bins": 80,
        "use_energy": False,
        "spec_aug": {
               "freq_mask_num": 2,
               "freq_mask_width": 27,
               "time_mask_num": 2,
               "time_mask_width": 100,
           }
        }
    fn = "file:testdata/100-121669-0000.wav"
    pipe = "pipe:flac -c -d -s testdata/103-1240-0005.flac |"
    sample_rate, waveform1 = utils.load_wave(fn)    
    sample_rate, waveform2 = utils.load_wave(pipe)
    waveform1 = torch.from_numpy(waveform1)
    waveform2 = torch.from_numpy(waveform2)
    lengths = [waveform1.shape[0], waveform2.shape[0]]
    max_length = max(lengths)
    print(lengths)
    padded_waveforms = torch.zeros(2, max_length)
    padded_waveforms[0, :lengths[0]] += waveform1
    padded_waveforms[1, :lengths[1]] += waveform2
    layer = SPLayer(conf)
    
    features, feature_lengths = layer(padded_waveforms, lengths)
   
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt    
    plt.imshow(features[1].numpy()) 
    plt.savefig("test.png")

    #print(features)
    #print(feature_lengths)


def specaug_test():
    featconf = {
        "feature_type": "fbank",
        "sample_rate": 16000,
        "num_mel_bins": 40,
        "use_energy": False
        }
    augconf = {
        "feature_type": "fbank",
        "sample_rate": 16000,
        "num_mel_bins": 40,
        "use_energy": False,
        "spec_aug": {
             "freq_mask_width": 10,
             "freq_mask_num": 2,
             "time_mask_width": 100,
             "time_mask_num": 2}
        }
    fn = "file:testdata/100-121669-0000.wav"
    pipe = "pipe:flac -c -d -s testdata/103-1240-0005.flac |"
    sample_rate, waveform1 = utils.load_wave(fn)    
    sample_rate, waveform2 = utils.load_wave(pipe)
    waveform1 = torch.from_numpy(waveform1)
    waveform2 = torch.from_numpy(waveform2)
    lengths = [waveform1.shape[0], waveform2.shape[0]]
    max_length = max(lengths)
    padded_waveforms = torch.zeros(2, max_length)
    padded_waveforms[0, :lengths[0]] += waveform1
    padded_waveforms[1, :lengths[1]] += waveform2
    splayer = SPLayer(featconf) 
    auglayer = SPLayer(augconf)
    features, feature_lengths = splayer(padded_waveforms, lengths)
    features2, feature_lengths2 = auglayer(padded_waveforms, lengths)
    print("Before augmentation")
    print(features)
    print("After augmentation")
    print(features2)

if __name__ == "__main__":
    fbank_test()
    specaug_test()
    specaug_fbank_test()




