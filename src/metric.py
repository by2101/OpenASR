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

import torch
import torch.nn as nn

import utils

class MetricSummarizer(object):
    def __init__(self):
        self.metrics = {} # name: (loss, weight)
        self.metric_names = []
        self.summarized = {}

    def register_metric(self, name, display=False, visual=False, optim=False):
        self.metric_names.append({
                "name": name,
                "display": display,
                "visual": visual,
                "optim": optim,
            }) 

    def reset_metrics(self):
        del self.metrics
        del self.summarized
        self.metrics = {} 
        for item in self.metric_names:
            self.metrics[item["name"]] = None
        self.summarized = {}
    
    def get_metric_by_name(self, name):
        return self.metrics[name]

    def update_metric(self, name, loss, weight=1.0):
        if name in self.metrics:
            self.metrics[name] = (loss, weight)
        else:
            raise ValueError("The metric {} is not registered.".format(name))

    def summarize(self):
        self.summarized = {} # name: torch.Tensor
        for key in self.metrics.keys():
            if self.metrics[key] is None:
                logging.warn("{} is not updated. Skip it.".format(key))
                continue
            item = self.metrics[key]
            self.summarized[key] = item[0] * item[1]

    def collect_loss(self):
        loss = 0
        for item in self.metric_names:
            key = item['name'] 
            if item["optim"] == True:
                v = self.metrics[key]
                loss += v[0] * v[1]
        return loss

    def fetch_scalers(self, use="display"):
        fetched = []
        for item in self.metric_names:
            if item[use] == True:
                if item["name"] not in self.summarized:
                    logging.warn("{} is not summarized. Skip it.".format(item["name"]))
                    continue
                fetched.append(
                    (item["name"], self.summarized[item["name"]]))
        return fetched
        
    def display_msg(self, fetched, max_item_one_line=3):
        msglist = []
        msglists = []
        cnt = 0
        for name, value in fetched:
            if isinstance(value, torch.Tensor):
                msglist.append("{}: {:.7f}".format(name, value.item()))
            else:
                msglist.append("{}: {:.7f}".format(name, value))
            cnt += 1
            if cnt == max_item_one_line:
                msglists.append(msglist)
                msglist = []
                cnt = 0
        if msglist:
            msglists.append(msglist)
        l = []
        for msglist in msglists:
            l.append(" | ".join(msglist))
        msg = "\n".join(l)
        return msg
    
    def visualize_scalers(self, fetched, step): 
        for name, value in fetched:
            if isinstance(value, torch.Tensor):
                utils.visualizer.add_scalar(name, value.item(), step)
            else:
                utils.visualizer.add_scalar(name, value, step)
