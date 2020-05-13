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
import time
import logging
import numpy as np
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
import torch.nn.init as init
import utils
import schedule
import metric


class Trainer(object):
    def __init__ (self, model, config, tr_loader, cv_loader):
        self.config = config
        self.tr_loader = tr_loader
        self.cv_loader = cv_loader

        self.model = model
        if config["multi_gpu"] == True:
            self.model_to_pack = self.model.module
        else:
            self.model_to_pack = self.model

        self.num_epoch = config["num_epoch"]
        self.exp_dir = config["exp_dir"]
        self.print_inteval = config["print_inteval"] 

        self.accumulate_grad_batch = config["accumulate_grad_batch"]
        self.init_lr = config["init_lr"]
        self.grad_max_norm = config["grad_max_norm"]
        self.label_smooth = config["label_smooth"]
        self.lst_w = 0.
        self.lst_t = 1.0
        if "lst" in config:
            self.lst_w = config["lst"]["lst_w"]
            self.lst_t = config["lst"]["lst_t"]

        self.num_last_ckpt_keep = None
        if "num_last_ckpt_keep" in config:
            self.num_last_ckpt_keep = config["num_last_ckpt_keep"]

        self.lr_scheduler = schedule.get_scheduler(config["lr_scheduler"]) 
        self.metric_summarizer = metric.MetricSummarizer()
        self.metric_summarizer.register_metric("per_token_loss", display=True, visual=True, optim=True)
        self.metric_summarizer.register_metric("avg_token_loss", display=True, visual=True, optim=False)
        self.metric_summarizer.register_metric("learning_rate", display=True, visual=True, optim=False)
        self.metric_summarizer.register_metric("sequence_per_sec", display=True, visual=True, optim=False)

        if utils.TENSORBOARD_LOGGING == 1:
            utils.visualizer.set_writer(os.path.join(self.exp_dir, "log"))
 
        # trainer state
        self.epoch = 0
        self.step = 0
        self.tr_loss = []
        self.cv_loss = []
        self.lr = self.init_lr
        
        if config["optimtype"] == "sgd": 
            self.optimizer = torch.optim.SGD(self.model_to_pack.parameters(), lr=self.lr, momentum=0.9)
        elif config["optimtype"] == "adam":
            self.optimizer = torch.optim.Adam(self.model_to_pack.parameters(), lr=self.lr, 
                betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        else:
            raise ValueError("Unknown optimizer.")
        if not os.path.isdir(self.exp_dir):
            os.makedirs(self.exp_dir)

        if utils.TENSORBOARD_LOGGING:
            (utts, padded_waveforms, wave_lengths, 
                    ids, labels, paddings) = next(iter(self.cv_loader)) # use a longer one
            if next(self.model_to_pack.parameters()).is_cuda:
                padded_waveforms = padded_waveforms.cuda()
                wave_lengths = wave_lengths.cuda()
                ids = ids.cuda()
                labels = labels.cuda()
                paddings = paddings.cuda()
            self.data_for_vis = (padded_waveforms, wave_lengths, ids, labels, paddings)         
    

    def training_state(self):
        return {
            "epoch": self.epoch,
            "step": self.step,
            "tr_loss": self.tr_loss,
            "cv_loss": self.cv_loss,
            "lr": self.lr,
            }    


    def restore_training_state(self, state):
        self.epoch = state["epoch"]
        self.step = state["step"]
        self.tr_loss = state["tr_loss"]
        self.cv_loss = state["cv_loss"]
        self.lr = state["lr"]


    def package(self):
        return {
            "model": self.model_to_pack.package(),
            "trainer_config": self.config,
            "trainer_state": self.training_state(),
            "optim_state": self.optimizer.state_dict(),
            "scheduler_state": self.lr_scheduler.pack_state()
            }


    def save(self, path):
        pkg = self.package()
        torch.save(pkg, path)
        logging.info("Saving model to {}".format(path))


    def restore(self, pkg):        
        self.restore_training_state(pkg["trainer_state"])
        self.optimizer.load_state_dict(pkg['optim_state'])
        self.lr_scheduler.restore_state(pkg["scheduler_state"])


    def train(self): 
        timer = utils.Timer()
        self.best_cvloss = 9e20
        if self.cv_loss:
            self.best_cvloss = min(self.cv_loss)

        if utils.TENSORBOARD_LOGGING == 1:
            self.visualize_figure()

        while self.epoch < self.num_epoch:
            timer.tic()
            self.epoch += 1
            logging.info("Training")
            tr_loss = self.iter_one_epoch()
            tr_msg = ("tr loss: {:.4f}").format(tr_loss)
            msg = "\n" + "-"*85 + "\n"
            msg += "Epoch {} Training Summary:\n{}\n".format(self.epoch, tr_msg)
            msg += "-"*85
            logging.info(msg)
            self.save(os.path.join(self.exp_dir, "ep-{:04d}.pt".format(self.epoch))) 
            self.save(os.path.join(self.exp_dir, "last-ckpt.pt")) 
            logging.info("Validation")
            cv_loss = self.iter_one_epoch(cross_valid=True)

            if self.best_cvloss > cv_loss:
                self.best_cvloss = cv_loss
            train_time = timer.toc()
            cv_msg = ("cv loss: {:.4f} | best cv loss {:.4f}").format(cv_loss, self.best_cvloss)
            msg = "\n" + "-"*85 + "\n"
            msg += "Epoch {} Validation Summary:\n{}\n".format(self.epoch, cv_msg)
            msg += "Time cost: {:.4f} min".format(train_time/60.)
            msg += "\n" + "-"*85 + '\n'
            logging.info(msg)
            if isinstance(self.lr_scheduler, schedule.BobLearningRateSchedule):
                self.lr_scheduler.update_decay_rate(cv_tot_loss/cv_utter_itered)
            self.tr_loss.append(tr_loss)
            self.cv_loss.append(cv_loss)

            if utils.TENSORBOARD_LOGGING == 1:
                utils.visualizer.add_scalar("tr_loss", tr_loss, self.epoch)
                utils.visualizer.add_scalar("cv_loss", cv_loss, self.epoch)
            if self.num_last_ckpt_keep:
                utils.cleanup_ckpt(self.exp_dir, self.num_last_ckpt_keep)


    def iter_one_epoch(self, cross_valid=False):
        niter = 0

        if cross_valid:
            loader = self.cv_loader
            self.model.eval()
        else:
            loader = self.tr_loader
            self.model.train()
        
        timer = utils.Timer()
        timer.tic()
        tot_loss = 0.
        tot_token = 0
        tot_sequence = 0
        tot_err_num = 0.
        tot_ref_num = 0.

        n_accu_batch = self.accumulate_grad_batch   

        loader_iter = iter(loader)
        tot_iter_num = len(loader_iter)
        while True:
            try:
                data = next(loader_iter)
                niter += 1
            except StopIteration:
                break
            (utts, padded_waveforms, wave_lengths, 
                    ids, labels, paddings) = data
            if cross_valid:
                with torch.no_grad():
                    this_loss = self.model(padded_waveforms.cuda(),
                            wave_lengths.cuda(),
                            ids.cuda(), 
                            labels.cuda(),
                            paddings.cuda())
            else:
                this_loss = self.model(padded_waveforms.cuda(),
                        wave_lengths.cuda(),
                        ids.cuda(), 
                        labels.cuda(),
                        paddings.cuda(),
                        label_smooth=self.label_smooth,
                        lst_w=self.lst_w,
                        lst_t=self.lst_t)

            batch_loss = torch.sum(this_loss)
            n_token = torch.sum(1-paddings).float()
            n_sequence = len(utts)

            tot_loss = tot_loss + batch_loss
            tot_token = tot_token + n_token
            tot_sequence = tot_sequence + n_sequence

            self.metric_summarizer.reset_metrics()
            self.metric_summarizer.update_metric("per_token_loss", batch_loss, 1.0/n_token)
            self.metric_summarizer.update_metric("avg_token_loss", tot_loss, 1.0/tot_token)
            self.metric_summarizer.update_metric("learning_rate", list(self.optimizer.param_groups)[0]["lr"], 1.0)
            self.metric_summarizer.update_metric("sequence_per_sec", tot_sequence, 1.0/timer.toc())
            self.metric_summarizer.summarize()

            loss =  self.metric_summarizer.collect_loss()
            loss = loss/self.accumulate_grad_batch
           
            # compute gradients 
            if not cross_valid:
                if n_accu_batch == self.accumulate_grad_batch:
                    self.optimizer.zero_grad()
                loss.backward()
                n_accu_batch -= 1
                if n_accu_batch == 0 or niter == tot_iter_num:
                    self.step += 1  # to be consistant with metric
                    grad_norm = clip_grad_norm_(self.model.parameters(), self.grad_max_norm)
                    self.lr_scheduler.step()   # then, update learning rate
                    self.lr_scheduler.set_lr(self.optimizer, self.init_lr)
                    self.optimizer.step() 
                    n_accu_batch = self.accumulate_grad_batch
                else:
                    continue

            if utils.TENSORBOARD_LOGGING == 1 and not cross_valid:
                tovis = self.metric_summarizer.fetch_scalers(use="visual")
                self.metric_summarizer.visualize_scalers(tovis, self.step)
                del tovis

 
            tot_time = timer.toc()
            if niter % self.print_inteval == 0:
                todisp = self.metric_summarizer.fetch_scalers(use="display")
                todispmsg = self.metric_summarizer.display_msg(todisp)
                del todisp
                msg = ("\nEpoch {} | Step {} | Iter {}:\n").format(self.epoch, self.step, niter)
                msg += todispmsg
                logging.info("Progress:\n" + msg.strip())

            if utils.TENSORBOARD_LOGGING == 1 and not cross_valid and self.step % 1000 == 0:
                self.visualize_figure()

        self.metric_summarizer.reset_metrics()
        torch.cuda.empty_cache()
        time.sleep(2)
        return (tot_loss/tot_token).item()


    def visualize_figure(self):
        with torch.no_grad():
            _, atten_info = self.model_to_pack(self.data_for_vis[0],
                    self.data_for_vis[1],
                    self.data_for_vis[2], 
                    self.data_for_vis[3],
                    self.data_for_vis[4],
                    return_atten=True)
        
        
        enc_length = atten_info[1][0]
        enc_output = atten_info[0][0].detach().cpu().numpy()[:enc_length, :]
        utils.visualizer.add_img_figure("enc_output", enc_output.transpose(), self.step)
        enc_self_att_probs = [t.detach().cpu().numpy()[0][:enc_length, :enc_length] 
                for t in atten_info[2]]
        tgt_length = atten_info[3][0]
        dec_self_att_probs = [t[0].detach().cpu().numpy()[0][:tgt_length, :tgt_length]
                for t in atten_info[4]]
        dec_enc_att_probs = [t[1].detach().cpu().numpy()[0][:tgt_length, :enc_length] 
                for t in atten_info[4]]
        for i, enc_self in enumerate(enc_self_att_probs):
            utils.visualizer.add_img_figure("enc_att_{}".format(i), enc_self, self.step)
        for i, (dec_self, dec_enc) in enumerate(zip(dec_self_att_probs, dec_enc_att_probs)):
            utils.visualizer.add_img_figure("dec_att_{}".format(i), dec_self, self.step)
            utils.visualizer.add_img_figure("dec_enc_att_{}".format(i), dec_enc, self.step)
        sp_output, sp_length = atten_info[5][0], atten_info[6][0]
        sp_output = sp_output.detach().cpu().numpy()[:sp_length, :]
        utils.visualizer.add_img_figure("sp_output", sp_output.transpose(), self.step)      


class LmTrainer(object):
    def __init__ (self, model, config, tr_loader, cv_loader):
        self.config = config
        self.tr_loader = tr_loader
        self.cv_loader = cv_loader
        self.model = model
        if config["multi_gpu"] == True:
            self.model_to_pack = self.model.module
        else:
            self.model_to_pack = self.model

        self.num_epoch = config["num_epoch"]
        self.exp_dir = config["exp_dir"]
        self.print_inteval = config["print_inteval"] 

        self.accumulate_grad_batch = config["accumulate_grad_batch"]
        self.init_lr = config["init_lr"]
        self.grad_max_norm = config["grad_max_norm"]
        self.label_smooth = config["label_smooth"]

        self.num_last_ckpt_keep = None
        if "num_last_ckpt_keep" in config:
            self.num_last_ckpt_keep = config["num_last_ckpt_keep"]

        self.lr_scheduler = schedule.get_scheduler(config["lr_scheduler"]) 
        self.metric_summarizer = metric.MetricSummarizer()
        self.metric_summarizer.register_metric("per_token_loss", display=True, visual=True, optim=True)
        self.metric_summarizer.register_metric("avg_token_loss", display=True, visual=True, optim=False)
        self.metric_summarizer.register_metric("per_token_acc", display=True, visual=True, optim=False)
        self.metric_summarizer.register_metric("avg_token_acc", display=True, visual=True, optim=False)
        self.metric_summarizer.register_metric("learning_rate", display=True, visual=True, optim=False)
        self.metric_summarizer.register_metric("token_per_sec", display=True, visual=True, optim=False)

        if utils.TENSORBOARD_LOGGING == 1:
            utils.visualizer.set_writer(os.path.join(self.exp_dir, "log"))
 
        # trainer state
        self.epoch = 0
        self.step = 0
        self.tr_loss = []
        self.cv_loss = []
        self.lr = self.init_lr
        
        if config["optimtype"] == "sgd": 
            self.optimizer = torch.optim.SGD(self.model_to_pack.parameters(), lr=self.lr, momentum=0.9)
        elif config["optimtype"] == "adam":
            self.optimizer = torch.optim.Adam(self.model_to_pack.parameters(), lr=self.lr, 
                betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        else:
            raise ValueError("Unknown optimizer.")

        if not os.path.isdir(self.exp_dir):
            os.makedirs(self.exp_dir)

        if utils.TENSORBOARD_LOGGING:
            (ids, labels, paddings) = next(iter(self.cv_loader)) # use a longer one
            if next(self.model_to_pack.parameters()).is_cuda:
                ids = ids.cuda()
                labels = labels.cuda()
                paddings = paddings.cuda()
            self.data_for_vis = ( ids, labels, paddings)         
 
    
    def training_state(self):
        return {
            "epoch": self.epoch,
            "step": self.step,
            "tr_loss": self.tr_loss,
            "cv_loss": self.cv_loss,
            "lr": self.lr,
            }    


    def restore_training_state(self, state):
        self.epoch = state["epoch"]
        self.step = state["step"]
        self.tr_loss = state["tr_loss"]
        self.cv_loss = state["cv_loss"]
        self.lr = state["lr"]


    def package(self):
        return {
            "model": self.model_to_pack.package(),
            "trainer_config": self.config,
            "trainer_state": self.training_state(),
            "optim_state": self.optimizer.state_dict(),
            "scheduler_state": self.lr_scheduler.pack_state()
            }


    def save(self, path):
        pkg = self.package()
        torch.save(pkg, path)
        logging.info("Saving model to {}".format(path))


    def restore(self, pkg):
        self.restore_training_state(pkg["trainer_state"])
        self.optimizer.load_state_dict(pkg['optim_state'])
        self.lr_scheduler.restore_state(pkg["scheduler_state"])


    def train(self): 
        timer = utils.Timer()
        self.best_cvloss = 9e20
        if self.cv_loss:
            self.best_cvloss = min(self.cv_loss)

        if utils.TENSORBOARD_LOGGING == 1 and self.config["vis_atten"]:
            self.visualize_figure()

        while self.epoch < self.num_epoch:
            timer.tic()
            self.epoch += 1
            logging.info("Training")
            tr_loss = self.iter_one_epoch()
            tr_msg = ("tr loss: {:.4f}").format(tr_loss)
            tr_msg += ", tr ppl {:.4f}".format(np.exp(tr_loss))
            msg = "\n" + "-"*85 + "\n"
            msg += "Epoch {} Training Summary:\n{}\n".format(self.epoch, tr_msg)
            msg += "-"*85
            logging.info(msg)
            self.save(os.path.join(self.exp_dir, "ep-{:04d}.pt".format(self.epoch))) 
            self.save(os.path.join(self.exp_dir, "last-ckpt.pt")) 
            logging.info("Validation")
            cv_loss = self.iter_one_epoch(cross_valid=True)
            if self.best_cvloss > cv_loss:
                self.best_cvloss = cv_loss
            train_time = timer.toc()
            cv_msg = ("cv loss: {:.4f} | best cv loss {:.4f} | ").format(cv_loss, self.best_cvloss)
            cv_msg += ("cv ppl: {:.4f} | best cv ppl {:.4f} | ").format(np.exp(cv_loss), np.exp(self.best_cvloss))
            msg = "\n" + "-"*85 + "\n"
            msg += "Epoch {} Validation Summary:\n{}\n".format(self.epoch, cv_msg)
            msg += "Time cost: {:.4f} min".format(train_time/60.)
            msg += "\n" + "-"*85 + '\n'
            logging.info(msg)
            if isinstance(self.lr_scheduler, schedule.BobLearningRateSchedule):
                self.lr_scheduler.update_decay_rate(np.exp(cv_loss))
            self.tr_loss.append(tr_loss)
            self.cv_loss.append(cv_loss)

            if utils.TENSORBOARD_LOGGING == 1:
                utils.visualizer.add_scalar("tr_loss/loss", tr_loss, self.epoch)
                utils.visualizer.add_scalar("cv_loss/loss", cv_loss, self.epoch)
                utils.visualizer.add_scalar("tr_loss/ppl", np.exp(tr_loss), self.epoch)
                utils.visualizer.add_scalar("cv_loss/ppl", np.exp(cv_loss), self.epoch)
         
            if self.num_last_ckpt_keep:
                utils.cleanup_ckpt(self.exp_dir, self.num_last_ckpt_keep)
 
        
    def iter_one_epoch(self, cross_valid=False):
        niter = 0

        if cross_valid:
            loader = self.cv_loader
            self.model.eval()
        else:
            loader = self.tr_loader
            self.model.train()
        
        timer = utils.Timer()
        timer.tic()
        tot_loss = 0.
        tot_token = 0
        tot_ncorrect = 0
        
        n_accu_batch = self.accumulate_grad_batch   

        loader_iter = iter(loader)
        tot_iter_num = len(loader_iter)
        while True:
            try:
                data = next(loader_iter)
                niter += 1
            except StopIteration:
                break
            (ids, labels, paddings) = data

            if cross_valid:
                with torch.no_grad():
                    this_loss, ncorrect = self.model(ids.cuda(),
                            labels.cuda(),
                            paddings.cuda(),
                            label_smooth=0)
 
            else: 
                this_loss, ncorrect = self.model(ids.cuda(),
                        labels.cuda(),
                        paddings.cuda(),
                        label_smooth=self.label_smooth)

            batch_loss = torch.sum(this_loss)
            batch_ncorrect = torch.sum(ncorrect)
            n_token = torch.sum(1-paddings).float()

            tot_loss = tot_loss + batch_loss
            tot_token = tot_token + n_token
            tot_ncorrect = tot_ncorrect + batch_ncorrect

            self.metric_summarizer.reset_metrics()
            self.metric_summarizer.update_metric("per_token_loss", batch_loss, 1.0/n_token)
            self.metric_summarizer.update_metric("avg_token_loss", tot_loss, 1.0/tot_token)
            self.metric_summarizer.update_metric("per_token_acc", batch_ncorrect, 1.0/n_token)
            self.metric_summarizer.update_metric("avg_token_acc", tot_ncorrect, 1.0/tot_token)
            self.metric_summarizer.update_metric("learning_rate", list(self.optimizer.param_groups)[0]["lr"], 1.0)
            self.metric_summarizer.update_metric("token_per_sec", tot_token, 1.0/timer.toc())
            self.metric_summarizer.summarize()

            loss =  self.metric_summarizer.collect_loss()
            loss = loss/self.accumulate_grad_batch
          
            # compute gradients 
            if not cross_valid:
                if n_accu_batch == self.accumulate_grad_batch:
                    self.optimizer.zero_grad()
                loss.backward()
                n_accu_batch -= 1
                if n_accu_batch == 0 or niter == tot_iter_num:
                    self.step += 1  # to be consistant with metric
                    grad_norm = clip_grad_norm_(self.model.parameters(), self.grad_max_norm)
                    self.lr_scheduler.step()   # then, update learning rate
                    self.lr_scheduler.set_lr(self.optimizer, self.init_lr)
                    self.optimizer.step() 
                    n_accu_batch = self.accumulate_grad_batch
                else:
                    continue

            if utils.TENSORBOARD_LOGGING == 1 and not cross_valid:
                tovis = self.metric_summarizer.fetch_scalers(use="visual")
                self.metric_summarizer.visualize_scalers(tovis, self.step)
                del tovis

            tot_time = timer.toc()
            if niter % self.print_inteval == 0:
                todisp = self.metric_summarizer.fetch_scalers(use="display")
                todispmsg = self.metric_summarizer.display_msg(todisp)
                del todisp
                msg = ("\nEpoch {} | Step {} | Iter {}:\n").format(self.epoch, self.step, niter)
                msg += todispmsg
                logging.info("Progress:\n" + msg.strip())

            if utils.TENSORBOARD_LOGGING == 1 and not cross_valid and self.step % 1000 == 0 and self.config["vis_atten"]:
                self.visualize_figure()

        self.metric_summarizer.reset_metrics()
        torch.cuda.empty_cache()
        time.sleep(2)
        return (tot_loss/tot_token).item()


    def visualize_figure(self):
        with torch.no_grad():
            atten_info = self.model_to_pack.fetch_vis_info(self.data_for_vis[0],
                    self.data_for_vis[1],
                    self.data_for_vis[2])
            enc_length = torch.sum(1-self.data_for_vis[2], dim=1).long()[0]
            enc_self_att_probs = [t.detach().cpu().numpy()[0][:enc_length, :enc_length] 
                    for t in atten_info]
            for i, enc_self in enumerate(enc_self_att_probs):
                utils.visualizer.add_img_figure("enc_att_{}".format(i), enc_self, self.step)
