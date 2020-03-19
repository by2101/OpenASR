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

import math
from collections import OrderedDict
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_
from third_party import transformer
import utils
import modules
from torch.nn.modules.normalization import LayerNorm


class Conv1dSubsample(torch.nn.Module):
    # the same as stack frames
    def __init__(self, input_dim, d_model, context_width, subsample):
        super(Conv1dSubsample, self).__init__()

        self.conv = nn.Conv1d(input_dim, d_model, context_width, stride=self.subsample)
        self.conv_norm = LayerNorm(self.d_model) 
        self.subsample = subsample
        self.context_width = context_width

    def forward(self, feats, feat_lengths): 
        outputs = self.conv(feats.permute(0, 2, 1))
        outputs = output.permute(0, 2, 1)
        outputs = self.conv_norm(outputs) 
        output_lengths = ((feat_lengths - 1*(self.context_width-1)-1)/self.subsample + 1).long()
        return outputs, output_lengths


class Conv2dSubsample(torch.nn.Module):
    # Follow ESPNet configuration
    def __init__(self, input_dim, d_model):
        super(Conv2dSubsample, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, 3, 2),
            torch.nn.ReLU()
        )            
        self.affine = torch.nn.Linear(32 * (((input_dim - 1) // 2 - 1) // 2), d_model)

    def forward(self, feats, feat_lengths): 
        outputs = feats.unsqueeze(1)  # [B, C, T, D]
        outputs = self.conv(outputs)
        B, C, T, D = outputs.size()
        outputs = outputs.permute(0, 2, 1, 3).contiguous().view(B, T, C*D)
        outputs = self.affine(outputs)
        output_lengths = (((feat_lengths-1) / 2 - 1) / 2).long()
        return outputs, output_lengths


class Conv2dSubsampleV2(torch.nn.Module):
    def __init__(self, input_dim, d_model, layer_num=2):
        super(Conv2dSubsampleV2, self).__init__()
        assert layer_num >= 1
        self.layer_num = layer_num
        layers = [("subsample/conv0", torch.nn.Conv2d(1, 32, 3, (2, 1))), 
                ("subsample/relu0", torch.nn.ReLU())]
        for i in range(layer_num-1):
            layers += [
                ("subsample/conv{}".format(i+1), torch.nn.Conv2d(32, 32, 3, (2, 1))),
                ("subsample/relu{}".format(i+1), torch.nn.ReLU())
            ] 
        layers = OrderedDict(layers)
        self.conv = torch.nn.Sequential(layers)            
        self.affine = torch.nn.Linear(32 * (input_dim-2*layer_num), d_model)

    def forward(self, feats, feat_lengths): 
        outputs = feats.unsqueeze(1)  # [B, C, T, D]
        outputs = self.conv(outputs)
        B, C, T, D = outputs.size()
        outputs = outputs.permute(0, 2, 1, 3).contiguous().view(B, T, C*D)
        outputs = self.affine(outputs)
        output_lengths = feat_lengths
        for _ in range(self.layer_num):
            output_lengths = ((output_lengths-1) / 2).long()
        return outputs, output_lengths


class Transformer(torch.nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()
        self.config = config     

        self.input_dim = config["input_dim"]
        self.d_model = config["d_model"]
        self.nhead = config["nhead"]
        self.dim_feedforward = config["dim_feedforward"]
        self.num_layers = config["num_layers"]
        self.dropout_rate = config["dropout_rate"]
        self.activation = config["activation"]
        self.subconf = config["sub"]
        if self.subconf["type"] == "ConvV1":
            self.sub = Conv2dSubsample(self.input_dim, self.d_model) 
        elif self.subconf["type"] == "ConvV2":
            self.sub = Conv2dSubsampleV2(self.input_dim, self.d_model, self.subconf["layer_num"]) 
        elif self.subconf["type"] == "Stack":
            self.context_width = config["context_width"]
            self.subsample = config["subsample"]
            self.sub = Conv1dSubsample(self.input_dim, self.d_model, self.context_width, self.subsample)
        
        self.scale = self.d_model ** 0.5

        self.pe = modules.PositionalEncoding(self.d_model)
        self.dropout = nn.Dropout(self.dropout_rate)
        encoder_norm = LayerNorm(self.d_model)
        encoder_layer = transformer.TransformerEncoderLayer(d_model=self.d_model, 
                nhead=self.nhead, dim_feedforward=self.dim_feedforward, 
                dropout=self.dropout_rate, activation=self.activation)
        self.transformer_encoder = transformer.TransformerEncoder(encoder_layer, self.num_layers, encoder_norm)

    def forward(self, feats, feat_lengths, return_atten=False):
        outputs, output_lengths = self.sub(feats, feat_lengths)
        outputs = self.dropout(self.pe(outputs))

        B, T, D_o = outputs.shape
        src_key_padding_mask = utils.get_transformer_padding_byte_masks(B, T, output_lengths).to(outputs.device)
        outputs = outputs.permute(1, 0, 2)
        if return_atten:
            outputs, self_atten_list = self.transformer_encoder(outputs, 
                    src_key_padding_mask=src_key_padding_mask, 
                    return_atten=True)
        else:
            outputs = self.transformer_encoder(outputs, 
                    src_key_padding_mask=src_key_padding_mask, 
                    return_atten=False)
        outputs = outputs.permute(1, 0, 2)
        if return_atten:
            return outputs, output_lengths, self_atten_list
        return outputs, output_lengths