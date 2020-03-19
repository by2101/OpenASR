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
import logging
import json
import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data.sampler import Sampler

import utils

IGNORE_ID = -1 

SOS_SYM = "<s>"
EOS_SYM = "</s>"
UNK_SYM = "<unk>"
SPECIAL_SYM_SET = {SOS_SYM, EOS_SYM, UNK_SYM}

class CharTokenizer(object):
    def __init__(self, fn_vocab):
        with open(fn_vocab, 'r') as f:
            units = f.read().strip().split('\n')
        units = [UNK_SYM, SOS_SYM, EOS_SYM] + units
        self.unit2id = {k:v for v,k in enumerate(units)}
        self.id2unit = units

    def to_id(self, unit):
        return self.unit2id[unit]

    def to_unit(self, id):
        return self.id2unit[id]

    def encode(self, textline):
        return [self.unit2id[char] 
            if char in self.unit2id 
            else self.unit2id[UNK_SYM] 
            for char in list(textline.strip())]

    def decode(self, ids, split_token=True, remove_special_sym=True):
        syms = [self.id2unit[i] for i in ids]
        if remove_special_sym:
            syms = [sym for sym in syms if sym not in SPECIAL_SYM_SET]
        if split_token:
            return " ".join(syms)
        return "".join(syms) 

    def unit_num(self):
        return len(self.unit2id)


def gen_casual_targets(idslist, maxlen, sos_id, eos_id):
    ids_with_sym_list = [[sos_id]+ids+[eos_id] for ids in idslist]
    B = len(idslist)
    padded_rawids = -torch.ones(B, maxlen+1).long()
 
    for b, ids in enumerate(ids_with_sym_list):
        if len(ids) > maxlen:
            logging.warn("ids length {} vs. maxlen {}, cut it.".format(len(ids), maxlen))
        l = min(len(ids), maxlen)
        padded_rawids[b, :l] = torch.tensor(ids).long()[:l]
    paddings = (padded_rawids==-1).long()
    padded_rawids = padded_rawids*(1-paddings) + eos_id*paddings # modify -1 to eos_id

    labels = padded_rawids[:, 1:]
    ids = padded_rawids[:, :-1]
    paddings = paddings[:, 1:] # the padding is for labels
    
    return ids, labels, paddings


class TextLineByLineDataset(data.Dataset):
    def __init__(self, fn):
        super(TextLineByLineDataset, self).__init__()
        with open(fn, 'r') as f:
            self.data = f.read().strip().split('\n') 

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class SpeechDataset(data.Dataset):
    def __init__(self, data_json_path, reverse=False):
        super(SpeechDataset, self).__init__()
        with open(data_json_path, 'rb') as f:
            data = json.load(f)
        self.data = sorted(data, key=lambda x: float(x["duration"]))
        if reverse:
            self.data.reverse()

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class KaldiDataset(data.Dataset):
    def __init__(self, data_dir, tag="file"):
        super(KaldiDataset, self).__init__()
        self.data = []
        if os.path.exists(os.path.join(data_dir, 'feats.scp')):
            p = os.path.join(data_dir, 'feats.scp')
        elif os.path.exists(os.path.join(data_dir, 'wav.scp')):
            p = os.path.join(data_dir, 'wav.scp')
        else:
            raise ValueError("None of feats.scp or wav.scp exist.")
        with open(p, 'r') as f:
            for line in f:
                utt, path = line.strip().split(' ', 1)
                path = "{}:{}".format(tag, path)
                d = (utt, path)
                self.data.append(d)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class TimeBasedSampler(Sampler): 
    def __init__(self, dataset, duration=200, ngpu=1, shuffle=False): # 200s
        self.dataset = dataset
        self.dur = duration
        self.shuffle = shuffle
        
        batchs = []
        batch = []
        batch_dur = 0.
        for idx in range(len(self.dataset)):
            batch.append(idx)
            batch_dur += self.dataset[idx]["duration"] 
            if batch_dur >= self.dur and len(batch)%ngpu==0: 
                # To make the numbers of batchs are equal for each GPU.
                batchs.append(batch)
                batch = []
                batch_dur = 0.
        if batch:
            if len(batch)%ngpu==0:
                batchs.append(batch)
            else:
                b = len(batch)
                batchs.append(batch[b//ngpu*ngpu:])
        self.batchs = batchs 

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.batchs)
        for b in self.batchs:
            yield b
         
    def __len__(self):
        return len(self.batchs)


def load_wave_batch(paths):
    waveforms = []
    lengths = []
    for path in paths:
        sample_rate, waveform = utils.load_wave(path) 
        waveform = torch.from_numpy(waveform)
        waveforms.append(waveform)
        lengths.append(waveform.shape[0])
    max_length = max(lengths)
    padded_waveforms = torch.zeros(len(lengths), max_length)
    for i in range(len(lengths)):
        padded_waveforms[i, :lengths[i]] += waveforms[i]
    return padded_waveforms, torch.tensor(lengths).long()


def load_feat_batch(paths):
    features = []
    lengths = []
    for path in paths:
        feature = utils.load_feat(path) 
        feature = torch.from_numpy(feature)
        features.append(feature)
        lengths.append(feature.shape[0])
    max_length = max(lengths)
    dim = feature.shape[1]
    padded_features = torch.zeros(len(lengths), max_length, dim)
    for i in range(len(lengths)):
        padded_features[i, :lengths[i], :] += features[i]
    return padded_features, torch.tensor(lengths).long()


class TextCollate(object):
    def __init__(self, tokenizer, maxlen):
        self.tokenizer = tokenizer
        self.maxlen = maxlen
        return

    def __call__(self, batch):
        timer = utils.Timer()        
        timer.tic()
        rawids_list = [self.tokenizer.encode(t) for t in batch]
        ids, labels, paddings = gen_casual_targets(rawids_list, self.maxlen, 
                self.tokenizer.to_id(SOS_SYM), self.tokenizer.to_id(EOS_SYM))
        logging.debug("Text Processing Time: {}s".format(timer.toc()))
        return ids, labels, paddings


class WaveCollate(object):
    def __init__(self, tokenizer, maxlen):
        self.tokenizer = tokenizer
        self.maxlen = maxlen
        return

    def __call__(self, batch):
        utts = [d["utt"] for d in batch] 
        paths = [d["path"] for d in batch] 
        trans = [d["transcript"] for d in batch] 
        timer = utils.Timer()        
        timer.tic()
        padded_waveforms, wave_lengths = load_wave_batch(paths)
        logging.debug("Wave Loading Time: {}s".format(timer.toc()))
        timer.tic()
        rawids_list = [self.tokenizer.encode(t) for t in trans]
        ids, labels, paddings = gen_casual_targets(rawids_list, self.maxlen, 
                self.tokenizer.to_id(SOS_SYM), self.tokenizer.to_id(EOS_SYM))
        logging.debug("Transcription Processing Time: {}s".format(timer.toc()))
        return utts, padded_waveforms, wave_lengths, ids, labels, paddings


class FeatureCollate(object):
    def __init__(self, tokenizer, maxlen):
        self.tokenizer = tokenizer
        self.maxlen = maxlen
        return

    def __call__(self, batch):
        utts = [d["utt"] for d in batch] 
        paths = [d["path"] for d in batch] 
        trans = [d["transcript"] for d in batch] 
        timer = utils.Timer()        
        timer.tic()
        padded_features, feature_lengths = load_feat_batch(paths)
        logging.debug("Feature Loading Time: {}s".format(timer.toc()))
        timer.tic()
        rawids_list = [self.tokenizer.encode(t) for t in trans]
        ids, labels, paddings = gen_casual_targets(rawids_list, self.maxlen, 
                self.tokenizer.to_id(SOS_SYM), self.tokenizer.to_id(EOS_SYM))
        logging.debug("Transcription Processing Time: {}s".format(timer.toc()))
        return utts, padded_features, feature_lengths, ids, labels, paddings


def kaldi_wav_collate(batch):
    utts = [d[0] for d in batch]
    paths = [d[1] for d in batch]
    padded_data, lengths = load_wave_batch(paths)
    return utts, padded_data, lengths


def kaldi_feat_collate(batch):
    utts = [d[0] for d in batch]
    paths = [d[1] for d in batch]
    padded_data, lengths = load_feat_batch(paths)
    return utts, padded_data, lengths
