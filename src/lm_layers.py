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

import logging
import math
import chardet
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn import init
from torch.nn.modules.normalization import LayerNorm

from third_party import transformer
import modules
import utils

class LSTM(nn.Module):
    def __init__(self, config):
        super(LSTM, self).__init__()
        self.config = config
        
        self.vocab_size = config["vocab_size"]
        self.hidden_size = config["hidden_size"]
        self.num_layers = config["num_layers"]
        self.dropout_rate = config["dropout_rate"]
        self.emb = nn.Embedding(self.vocab_size, self.hidden_size)       
        self.rnn = nn.LSTM(self.hidden_size, self.hidden_size, num_layers=self.num_layers, dropout=self.dropout_rate, batch_first=True)
        self.dropout1 = nn.Dropout(self.dropout_rate) 
        self.dropout2 = nn.Dropout(self.dropout_rate) 
        self.output_affine = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        self.emb.weight = self.output_affine.weight
        
    def forward(self, ids, lengths=None):
        outputs = self.emb(ids)
        outputs = self.dropout1(outputs)
        outputs, (h, c) = self.rnn(outputs) 
        outputs = self.dropout2(outputs)
        outputs = self.output_affine(outputs)
        return outputs

    def reset_parameters(self):
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
        init.uniform_(self.emb.weight, a=-0.01, b=0.01)


class TransformerLM(nn.Module):
    def __init__(self, config):
        super(TransformerLM, self).__init__()
        self.config = config
        
        self.vocab_size = config["vocab_size"]
        self.d_model = config["d_model"]
        self.nhead = config["nhead"]
        self.num_layers = config["num_layers"]
        self.dim_feedforward = config["dim_feedforward"]
        self.activation = config["activation"]
        self.dropout_rate = config["dropout_rate"]


        self.dropout = nn.Dropout(self.dropout_rate) 
        self.scale = self.d_model ** 0.5
        self.pe = modules.PositionalEncoding(self.d_model)
        self.emb = nn.Embedding(self.vocab_size, self.d_model)       
        encoder_layer = transformer.TransformerEncoderLayer(d_model=self.d_model, 
                nhead=self.nhead, dim_feedforward=self.dim_feedforward, 
                dropout=self.dropout_rate, activation=self.activation)
        self.transformer_encoder = transformer.TransformerEncoder(encoder_layer, self.num_layers)
        self.output_affine = nn.Linear(self.d_model, self.vocab_size, bias=False)
        self.emb.weight = self.output_affine.weight
        
    def forward(self, ids, lengths, return_atten=False):
        B, T = ids.shape 

        key_padding_mask = utils.get_transformer_padding_byte_masks(
            B, T, lengths).to(ids.device)
        casual_masks = utils.get_transformer_casual_masks(T).to(ids.device)

        outputs = self.emb(ids) * self.scale      
        outputs = self.pe(outputs)
        outputs = self.dropout(outputs)
        outputs = outputs.permute(1, 0, 2)

        outputs, self_atten_list = self.transformer_encoder(outputs,
                mask=casual_masks, 
                src_key_padding_mask=key_padding_mask, 
                return_atten=True)        
        outputs = self.output_affine(outputs)
        if return_atten: 
            return outputs, self_atten_list
        return outputs


