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
import torch
import math


def get_scheduler(config):
    if config["type"] == "linear":
        return LinearLearningRateSchedule(config)
    elif config["type"] == "warmup_linear":
        return WarmupLinearLearningRateSchedule(config)
    elif config["type"] == "bob":
        return BobLearningRateSchedule(config)
    elif config["type"] == "warmup_transformer":
        return WarmupTransformerLearningRateSchedule(config)
    else:
        raise ValueError("Unknown scheduler.")


class BaseLearningRateSchedule(object): 
    def __init__(self):
        self.step_num = 0
        self.decay_rate = 1.
        self.config = None
        self.misc_state = -1
        self.update_only_with_step = True

    def set_lr(self, optimizer, init_lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = init_lr * self.decay_rate

    def step(self):
        self.step_num += 1
        if self.update_only_with_step:
            self.update_decay_rate()    
            
    def pack_state(self):
        pkg = {
            "step": self.step_num,
            "decay_rate": self.decay_rate,
            "misc_state": self.misc_state
            }
        return pkg
        
    def restore_state(self, pkg):
        self.step_num = pkg['step']
        self.decay_rate = pkg['decay_rate']
        self.misc_state = pkg['misc_state'] 
        self.check_misc_state()

    def check_misc_state(self):
        raise NotImplementedError()

    def update_decay_rate(self):
        raise NotImplementedError()
    
    
def compute_polynomial_intep(x, x0, y0, x1, y1, power):
    if x < x0:
        return y0
    elif x > x1:
        return y1
    else:
        if power != 1.0:
            f = ((1.0 * x - x0) / (x1 - x0)) ** power      
        else:
            f = ((1.0 * x - x0) / (x1 - x0))
        y = y0 + f * (y1 - y0)
        return y


def compute_linear_intep(x, x0, y0, x1, y1):
    return compute_polynomial_intep(x, x0, y0, x1, y1, 1.0)


class LinearLearningRateSchedule(BaseLearningRateSchedule):    
    def __init__(self, conf):
        super(LinearLearningRateSchedule, self).__init__()
        self.config = {
                "x0": conf["x0"],
                "y0": conf["y0"],
                "x1": conf["x1"],
                "y1": conf["y1"],
            }
    def check_misc_state(self):
        pass

   
    def update_decay_rate(self):
        self.decay_rate = compute_linear_intep(self.step_num, self.config["x0"], 
            self.config["y0"], self.config["x1"], self.config["y1"])    

         
class WarmupLinearLearningRateSchedule(LinearLearningRateSchedule):    
    def __init__(self, conf):
        super(WarmupLinearLearningRateSchedule, self).__init__(conf)
        self.config["warmup_step"] = conf["warmup_step"]
                
    def update_decay_rate(self):
        dc0 = compute_linear_intep(self.step_num, 0, 
            0, self.config["warmup_step"], self.config["y0"])
        dc1 = compute_linear_intep(self.step_num, self.config["x0"], 
            self.config["y0"], self.config["x1"], self.config["y1"])
        self.decay_rate = min(dc0, dc1)  


class WarmupTransformerLearningRateSchedule(BaseLearningRateSchedule):    
    def __init__(self, conf):
        super(WarmupTransformerLearningRateSchedule, self).__init__()
        self.config = {}
        self.config["warmup_step"] = conf["warmup_step"]
        self.config["d_model"] = conf["d_model"]      

    def update_decay_rate(self):
        d0 = self.step_num**(-0.5)
        d1 = self.step_num*(self.config["warmup_step"]**(-1.5))
        self.decay_rate = (self.config["d_model"]**(-0.5))*min(d0, d1)  

    def check_misc_state(self):
        pass


class BobLearningRateSchedule(BaseLearningRateSchedule):    
    def __init__(self, conf):
        super(BobLearningRateSchedule, self).__init__()
        self.update_only_with_step = False
        self.config = {
                "decay_coef": conf["decay_coef"],
                "tolerate": conf["tolerate"],
            }
        self.misc_state = {
                "last_loss": -1,
                "last_decay_rate": 1,
            }
    
    def update_decay_rate(self, this_loss): 
        improvement = (self.misc_state["last_loss"] - this_loss)/self.misc_state["last_loss"]
        if improvement < self.config["tolerate"]:
            logging.info(("Improvment {:.4f} is smaller than tolerate {:.4f},"
                " decay LR.").format(improvement, self.config["tolerate"]))
            new_decay_rate = self.misc_state["last_decay_rate"] * self.config["decay_coef"] 
            self.decay_rate = new_decay_rate
            self.misc_state["last_decay_rate"] = new_decay_rate 
        self.misc_state["last_loss"] = this_loss     

    def check_misc_state(self):
        if (not "last_loss" in self.misc_state or  
            not "last_decay_rate" in self.misc_state):
            raise ValueError("The misc states are not match. Maybe the package was not trained with Bob lr schedule.")
