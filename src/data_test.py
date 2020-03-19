"""
Copyright 2020 Ye Bai by1993@qq.com

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import sys
import numpy as np
import torch.utils.data
import data
import pdb

os.chdir(os.path.abspath(os.path.dirname(__file__)))

def test_dataset():
    dataset = data.SpeechDataset("testdata/test.json")
    print(dataset[0])

def test_dataloader():
    dataset = data.SpeechDataset("testdata/test.json")
    sampler = data.TimeBasedSampler(dataset, 5)
    tokenizer = data.CharTokenizer("testdata/train_chars.txt")
    collate = data.WaveCollate(tokenizer, 60)
    dataloader = torch.utils.data.DataLoader(dataset, 
            batch_sampler=sampler, collate_fn=collate, shuffle=False)
    dataiter = iter(dataloader)
    batch = next(dataiter)
    utts, padded_waveforms, wave_lengths, ids, labels, paddings = batch

    print(utts[0])
    print(ids[0])
    print(labels[0])
    print(paddings[0])

 
if __name__ == "__main__":
    test_dataloader()
    test_dataset()
