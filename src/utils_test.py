import os
import sys
import subprocess
import numpy as np
import torch

import utils

os.chdir(os.path.abspath(os.path.dirname(__file__)))

def test_cleanup():
    expdir = "testdata/cleanup"
    os.makedirs(expdir)
    for i in range(120):
        with open(os.path.join(expdir, "ckpt-{:04d}.pt".format(i)), 'w') as f:
            f.write("")
    with open(os.path.join(expdir, "last-ckpt.pt"), 'w') as f:
            f.write("")
    utils.cleanup_ckpt(expdir, 3)
    len(os.listdir(expdir)) == 4


def test_read_wave_from_pipe():
    command = "flac -c -d -s testdata/100-121669-0000.flac " 
    output = utils.get_command_stdout(command)
    with open("testdata/100-121669-0000.wav", 'rb') as f:
        wav_content = f.read()
    assert output == wav_content

def test_load_wave():
    pipe = "pipe:flac -c -d -s testdata/100-121669-0000.flac | "
    fn = "file:testdata/100-121669-0000.wav"
    ark = "ark:/data1/Corpora/LibriSpeech/ark/train_960.ark:16"
    ark2 = "ark:/data1/Corpora/LibriSpeech/ark/train_960.ark:2591436"
    timer = utils.Timer()
    timer.tic()
    s3, d3 = utils.load_wave(ark)
    print("Load ark time: {}s".format(timer.toc()))
    timer.tic()
    s2, d2 = utils.load_wave(fn)
    print("Load file time: {}s".format(timer.toc()))
    timer.tic()
    s1, d1 = utils.load_wave(pipe)
    print("Load flac pipe time: {}s".format(timer.toc()))
    print("Load ark2")
    
    s, d = utils.load_wave(ark2)
    
    assert s1 == s2
    assert s3 == s2
    assert np.sum(d1!=d2) == 0
    assert np.sum(d3!=d2) == 0

def test_get_transformer_casual_masks():
    print('test_get_transformer_casual_masks')
    print(utils.get_transformer_casual_masks(5))

def test_get_transformer_padding_byte_masks():
    B = 3
    T = 5
    lengths = torch.tensor([3, 4, 5]).long()
    masks = utils.get_transformer_padding_byte_masks(B, T, lengths)
    print('test_get_transformer_padding_byte_masks')
    print(masks)

if __name__ == "__main__":
    test_cleanup()
    test_read_wave_from_pipe()
    test_load_wave()
    test_get_transformer_casual_masks()
    test_get_transformer_padding_byte_masks()



