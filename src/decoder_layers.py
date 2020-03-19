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
import torch.nn.functional as F
from torch.nn import init
from torch.nn.modules.normalization import LayerNorm

from third_party import transformer
import utils
import modules


class TransformerDecoder(nn.Module):
    def __init__(self, config):
        super(TransformerDecoder, self).__init__()
        self.config = config
        
        self.d_model = config["d_model"]
        self.nhead = config["nhead"]
        self.num_layers = config["num_layers"]
        self.encoder_dim = config["encoder_dim"]
        self.dim_feedforward = config["dim_feedforward"] 
        self.vocab_size = config["vocab_size"]
        self.dropout_rate = config["dropout_rate"]
        self.activation = config["activation"]

        self.emb = nn.Embedding(self.vocab_size, self.d_model)       
        self.emb_scale = self.d_model ** 0.5
        self.pe = modules.PositionalEncoding(self.d_model)
        self.dropout = nn.Dropout(self.dropout_rate) 
      
        transformer_decoder_layer = transformer.TransformerDecoderLayer(
                d_model=self.d_model, 
                nhead=self.nhead, 
                dim_feedforward=self.dim_feedforward,
                dropout=self.dropout_rate,
                activation=self.activation)
        self.transformer_block = transformer.TransformerDecoder(transformer_decoder_layer, 
                self.num_layers)

        self.output_affine = nn.Linear(self.d_model, self.vocab_size)
        nn.init.xavier_normal_(self.output_affine.weight)
        self.emb.weight = self.output_affine.weight # tying weight

    def forward(self, encoder_outputs, encoder_output_lengths, decoder_inputs, decoder_input_lengths, return_atten=False):

        B, T_e, D_e = encoder_outputs.shape
        encoder_outputs = encoder_outputs.permute(1, 0, 2) # [S, B, D_e]

        _, T_d = decoder_inputs.shape 

        memory_key_padding_mask = utils.get_transformer_padding_byte_masks(
            B, T_e, encoder_output_lengths).to(encoder_outputs.device)
        tgt_key_padding_mask = utils.get_transformer_padding_byte_masks(
            B, T_d, decoder_input_lengths).to(encoder_outputs.device)
        casual_masks = utils.get_transformer_casual_masks(T_d).to(encoder_outputs.device)

        outputs = self.emb(decoder_inputs) * self.emb_scale      
        outputs = self.pe(outputs)
        outputs = self.dropout(outputs)
        outputs = outputs.permute(1, 0, 2)
        
        if return_atten:
            outputs, decoder_atten_tuple_list = self.transformer_block(outputs, encoder_outputs,
                memory_mask=None, memory_key_padding_mask=memory_key_padding_mask,
                tgt_key_padding_mask=tgt_key_padding_mask, tgt_mask=casual_masks,
                return_atten=True)        
        else:
            outputs = self.transformer_block(outputs, encoder_outputs,
                memory_mask=None, memory_key_padding_mask=memory_key_padding_mask,
                tgt_key_padding_mask=tgt_key_padding_mask, tgt_mask=casual_masks, 
                return_atten=False)
        outputs = outputs.permute(1, 0, 2)
        outputs = self.output_affine(outputs)

        if return_atten:
            return outputs, decoder_atten_tuple_list
        return outputs
