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
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_
import utils
import lm_layers
import pdb


class Model(torch.nn.Module):
    def __init__(self, splayer, encoder, decoder, lm=None):
        super(Model, self).__init__()
        self.splayer = splayer
        self.encoder = encoder
        self.decoder = decoder
        self._reset_parameters()
        
        self.lm = lm  # this must be set after parameter initialization


    def forward(self, batch_wave, lengths, target_ids, target_labels=None, target_paddings=None, label_smooth=0., lst_w=0., lst_t=1.0, return_atten=False):
        target_lengths = torch.sum(1-target_paddings, dim=-1).long()
        logits, atten_info = self.get_logits(batch_wave, lengths, 
                target_ids, target_lengths, return_atten=True)
        losses = self._compute_cross_entropy_losses(logits, target_labels, target_paddings)
        loss = torch.sum(losses)
        if label_smooth > 0:
            loss = loss*(1-label_smooth) + self._uniform_label_smooth(logits, target_paddings)*label_smooth
        if lst_w > 0.:
            loss = loss*(1-lst_w) + self._lst(logits, target_ids, target_paddings, T=lst_t)*lst_w
        if return_atten:
            return loss, atten_info
        return loss


    def _uniform_label_smooth(self, logits, paddings):
        log_probs = F.log_softmax(logits, dim=-1)
        nlabel = log_probs.shape[-1]
        ent_uniform = -torch.sum(log_probs, dim=-1)/nlabel
        return torch.sum(ent_uniform*(1-paddings).float())


    def _lst(self, logits, target_ids, target_paddings, T=5.0): 
        with torch.no_grad(): 
            self.lm.eval()
            lengths = torch.sum(1-target_paddings, dim=-1).long()
            teacher_probs = self.lm.get_probs(target_ids, lengths, T=T)
        logprobs = torch.log_softmax(logits, dim=-1)
        losses = -torch.sum(teacher_probs * logprobs, dim=-1)
        return torch.sum(losses*(1-target_paddings).float())


    def _compute_cross_entropy_losses(self, logits, labels, paddings):
        B, T, V = logits.shape
        losses = F.cross_entropy(logits.view(-1, V), labels.view(-1), reduction="none").view(B, T) * (1-paddings).float()
        return losses


    def _compute_wers(self, hyps, labels):
        raise NotImplementedError()


    def _sample_nbest(self, encoder_output, encoder_output_lengths, nbest_keep=4,):
        self._beam_search(encoder_outputs, encoder_output_lengths, nbest_keep, sosid, maxlen)       
        raise NotImplementedError()


    def _compute_mwer_loss(self):
        raise NotImplementedError()


    def get_logits(self, batch_wave, lengths, target_ids, target_lengths, return_atten=False):
        if return_atten:
            timer = utils.Timer()
            timer.tic()
            sp_outputs, sp_output_lengths = self.splayer(batch_wave, lengths)
            logging.debug("splayer time: {}s".format(timer.toc()))
            timer.tic()
            encoder_outputs, encoder_output_lengths, enc_self_atten_list = self.encoder(sp_outputs, sp_output_lengths, return_atten=True)
            logging.debug("encoder time: {}s".format(timer.toc()))
            timer.tic()
            outputs, decoder_atten_tuple_list = self.decoder(encoder_outputs, encoder_output_lengths, target_ids, target_lengths, return_atten=True)
            logging.debug("decoder time: {}s".format(timer.toc()))
            timer.tic()
            return outputs, (encoder_outputs, encoder_output_lengths, enc_self_atten_list, target_lengths, decoder_atten_tuple_list, sp_outputs, sp_output_lengths)
        else:
            timer = utils.Timer()
            timer.tic()
            encoder_outputs, encoder_output_lengths = self.splayer(batch_wave, lengths)
            logging.debug("splayer time: {}s".format(timer.toc()))
            timer.tic()
            encoder_outputs, encoder_output_lengths = self.encoder(encoder_outputs, encoder_output_lengths, return_atten=False)
            logging.debug("encoder time: {}s".format(timer.toc()))
            timer.tic()
            outputs = self.decoder(encoder_outputs, encoder_output_lengths, target_ids, target_lengths, return_atten=False)
            logging.debug("decoder time: {}s".format(timer.toc()))
            timer.tic()
            return outputs


    def decode(self, batch_wave, lengths, nbest_keep, sosid=1, eosid=2, maxlen=100):
        if type(nbest_keep) != int:
            raise ValueError("nbest_keep must be a int.")
        encoder_outputs, encoder_output_lengths = self._get_acoustic_representations(
                batch_wave, lengths)
        target_ids, scores = self._beam_search(encoder_outputs, encoder_output_lengths, nbest_keep, sosid, eosid, maxlen) 
        return target_ids, scores


    def _get_acoustic_representations(self, batch_wave, lengths):
        encoder_outputs, encoder_output_lengths = self.splayer(batch_wave, lengths)
        encoder_outputs, encoder_output_lengths = self.encoder(encoder_outputs, encoder_output_lengths, return_atten=False)
        return encoder_outputs, encoder_output_lengths


    def _beam_search(self, encoder_outputs, encoder_output_lengths, nbest_keep, sosid, eosid, maxlen):

        B = encoder_outputs.shape[0] 
        # init
        init_target_ids = torch.ones(B, 1).to(encoder_outputs.device).long()*sosid
        init_target_lengths = torch.ones(B).to(encoder_outputs.device).long()
        outputs = (self.decoder(encoder_outputs, encoder_output_lengths, init_target_ids, init_target_lengths)[:, -1, :])
        vocab_size = outputs.size(-1)
        outputs = outputs.view(B, vocab_size)
        log_probs = F.log_softmax(outputs, dim=-1)
        topk_res = torch.topk(log_probs, k=nbest_keep, dim=-1) 
        nbest_ids = topk_res[1].view(-1)  #[batch_size*nbest_keep, 1]
        nbest_logprobs = topk_res[0].view(-1)

        target_ids = torch.ones(B*nbest_keep, 1).to(encoder_outputs.device).long()*sosid
        target_lengths = torch.ones(B*nbest_keep).to(encoder_outputs.device).long()

        target_ids = torch.cat([target_ids, nbest_ids.view(B*nbest_keep, 1)], dim=-1)
        target_lengths += 1

        finished_sel = None
        ended = []        
        ended_scores = []
        ended_batch_idx = []
        for step in range(1, maxlen):
            (nbest_ids, nbest_logprobs, beam_from) = self._decode_single_step(
                    encoder_outputs, encoder_output_lengths, target_ids, target_lengths, nbest_logprobs, finished_sel)
            batch_idx = (torch.arange(B)*nbest_keep).view(B, -1).repeat(1, nbest_keep).contiguous().to(beam_from.device)
            batch_beam_from = (batch_idx + beam_from.view(-1, nbest_keep)).view(-1)
            nbest_logprobs = nbest_logprobs.view(-1)
            finished_sel = (nbest_ids.view(-1) == eosid)
            target_ids = target_ids[batch_beam_from]
            target_ids = torch.cat([target_ids, nbest_ids.view(B*nbest_keep, 1)], dim=-1)
            target_lengths += 1

            for i in range(finished_sel.shape[0]): 
                if finished_sel[i]:
                    ended.append(target_ids[i])
                    ended_scores.append(nbest_logprobs[i])
                    ended_batch_idx.append(i // nbest_keep)
            target_ids = target_ids * (1 - finished_sel[:, None].long()) # mask out finished

        for i in range(target_ids.shape[0]): 
            ended.append(target_ids[i])
            ended_scores.append(nbest_logprobs[i])
            ended_batch_idx.append(i // nbest_keep)

        formated = {}
        for i in range(B):
            formated[i] = []
        for i in range(len(ended)):
            if ended[i][0] == sosid:
                formated[ended_batch_idx[i]].append((ended[i], ended_scores[i]))
        for i in range(B):
            formated[i] = sorted(formated[i], key=lambda x:x[1], reverse=True)[:nbest_keep]
       
        target_ids = torch.zeros(B, nbest_keep, maxlen+1).to(encoder_outputs.device).long() 
        scores = torch.zeros(B, nbest_keep).to(encoder_outputs.device)
        for i in range(B):
            for j in range(nbest_keep):
                item = formated[i][j]
                l = min(item[0].shape[0], target_ids[i, j].shape[0])
                target_ids[i, j, :l] = item[0][:l]
                scores[i, j] = item[1]
        return target_ids, scores


    def _decode_single_step(self, encoder_outputs, encoder_output_lengths, target_ids, target_lengths, accumu_scores, finished_sel=None):
        """
        encoder_outputs: [B, T_e, D_e]
        encoder_output_lengths: [B]
        target_ids: [B*nbest_keep, T_d]
        target_lengths: [B*nbest_keep]
        accumu_scores: [B*nbest_keep]
        """

        B, T_e, D_e = encoder_outputs.shape 
        B_d, T_d = target_ids.shape
        if B_d % B != 0:
            raise ValueError("The dim of target_ids does not match the encoder_outputs.")
        nbest_keep = B_d // B
        encoder_outputs = (encoder_outputs.view(B, 1, T_e, D_e)
                .repeat(1, nbest_keep, 1, 1).view(B*nbest_keep, T_e, D_e))
        encoder_output_lengths = (encoder_output_lengths.view(B, 1)
                .repeat(1, nbest_keep).view(-1))

        # outputs: [B, nbest_keep, vocab_size]
        outputs = (self.decoder(encoder_outputs, encoder_output_lengths, target_ids, target_lengths)[:, -1, :])
        vocab_size = outputs.size(-1)
        outputs = outputs.view(B, nbest_keep, vocab_size)
        log_probs = F.log_softmax(outputs, dim=-1)  # [B, nbest_keep, vocab_size]
        if finished_sel is not None:
            log_probs = log_probs.view(B*nbest_keep, -1) - finished_sel.view(B*nbest_keep, -1).float()*9e9
            log_probs = log_probs.view(B, nbest_keep, vocab_size)
        this_accumu_scores = accumu_scores.view(B, nbest_keep, 1) + log_probs 
        topk_res = torch.topk(this_accumu_scores.view(B, nbest_keep*vocab_size), k=nbest_keep, dim=-1) 

        nbest_logprobs = topk_res[0]  # [B, nbest_keep]
        nbest_ids = topk_res[1] % vocab_size # [B, nbest_keep]
        beam_from = (topk_res[1] / vocab_size).long()
        return nbest_ids, nbest_logprobs, beam_from


    def package(self):
        pkg = {
            "splayer_config": self.splayer.config,
            "splayer_state": self.splayer.state_dict(),
            "encoder_config": self.encoder.config,
            "encoder_state": self.encoder.state_dict(),
            "decoder_config": self.decoder.config,
            "decoder_state": self.decoder.state_dict(),
             }
        return pkg


    def restore(self, pkg):
        # check config
        logging.info("Restore model states...")
        for key in self.splayer.config.keys():
            if key == "spec_aug":
                continue
            if self.splayer.config[key] != pkg["splayer_config"][key]:
                raise ValueError("splayer_config mismatch.")
        for key in self.encoder.config.keys():
            if (key != "dropout_rate" and 
                    self.encoder.config[key] != pkg["encoder_config"][key]):
                raise ValueError("encoder_config mismatch.")
        for key in self.decoder.config.keys():
            if (key != "dropout_rate" and 
                    self.decoder.config[key] != pkg["decoder_config"][key]):
                raise ValueError("decoder_config mismatch.")
 
        self.splayer.load_state_dict(pkg["splayer_state"])
        self.encoder.load_state_dict(pkg["encoder_state"])
        self.decoder.load_state_dict(pkg["decoder_state"])
     

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)  


class LM(torch.nn.Module):
    def __init__(self, lmlayer):
        super(LM, self).__init__()
        self.lm_layer = lmlayer
        #self._reset_parameters()
        

    def forward(self, ids, labels, paddings, label_smooth=0.): 
        lengths = torch.sum(1-paddings, dim=1).long()
        logits = self.get_logits(ids, lengths)
        ntoken = torch.sum(1-paddings)
        tot_loss = torch.sum(self._compute_ce_loss(logits, labels, paddings))
        if label_smooth > 0:
            tot_loss = tot_loss*(1-label_smooth) + self._uniform_label_smooth(logits, paddings)*label_smooth
        tot_ncorrect = self._compute_ncorrect(logits, labels, paddings)
        return tot_loss, tot_ncorrect


    def fetch_vis_info(self, ids, labels, paddings):
        lengths = torch.sum(1-paddings, dim=1).long()
        atten = None
        if isinstance(self.lm_layer, lm_layers.TransformerLM):
            logits, atten = self.lm_layer(ids, lengths, return_atten=True)
        elif (isinstance(self.lm_layer, clozer.ClozerV2) or
                isinstance(self.lm_layer, clozer.Clozer) or
                isinstance(self.lm_layer, clozer.UniClozer) or
                isinstance(self.lm_layer, clozer.BwdUniClozer)):
            logits, atten = self.lm_layer(ids, lengths, return_atten=True)
        else:
            raise ValueError('Unknown lm layer')
        return atten
    

    def get_probs(self, ids, lengths, T=1.0):
        logits = self.get_logits(ids, lengths)
        probs = F.softmax(logits/T, dim=-1)
        return probs


    def get_logprobs(self, ids, lengths, T=1.0):
        logits = self.get_logits(ids, lengths)
        logprobs = F.log_softmax(logits/T, dim=-1)
        return logprobs


    def get_logits(self, ids, lengths=None):
        if len(ids.shape) == 1:
            B = ids.shape[0]
            ids = ids.view(B, 1).contiguous()
        logits = self.lm_layer(ids, lengths)
        return logits


    def _compute_ce_loss(self, logits, labels, paddings):
        D = logits.size(-1)
        losses = F.cross_entropy(logits.view(-1, D).contiguous(), labels.view(-1), reduction='none')
        return losses * (1-paddings).view(-1).float()


    def _uniform_label_smooth(self, logits, paddings):
        log_probs = F.log_softmax(logits, dim=-1)
        nlabel = log_probs.shape[-1]
        ent_uniform = -torch.sum(log_probs, dim=-1)/nlabel
        return torch.sum(ent_uniform*(1-paddings).float())


    def _compute_ncorrect(self, logits, labels, paddings):
        D = logits.size(-1)
        logprobs = F.log_softmax(logits, dim=-1)
        pred = torch.argmax(logprobs.view(-1, D), dim=-1)
        n_correct = torch.sum((pred == labels.view(-1)).float() * (1-paddings).view(-1).float())
        return n_correct


    def package(self):
        pkg = {
            "lm_config": self.lm_layer.config,
            "lm_state": self.lm_layer.state_dict(),
             }
        return pkg


    def restore(self, pkg):
        # check config
        logging.info("Restore model states...")
        for key in self.lm_layer.config.keys():
            if (key != "dropout_rate" and 
                    self.lm_layer.config[key] != pkg["lm_config"][key]):
                raise ValueError("lm_config mismatch.")
 
        self.lm_layer.load_state_dict(pkg["lm_state"])
     
     
    def _reset_parameters(self):
        self.lm_layer.reset_parameters()
