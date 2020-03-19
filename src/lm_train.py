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

import sys
import os
import argparse
import logging
import yaml
import torch

if "LAS_LOG_LEVEL" in os.environ:
    LOG_LEVEL = os.environ["LAS_LOG_LEVEL"]
else:
    LOG_LEVEL = "INFO" 
if LOG_LEVEL == "DEBUG":
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')
else:
     logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')

import utils
import data
import lm_layers
import models
from trainer import LmTrainer


def get_args():
    parser = argparse.ArgumentParser(description="""
     Usage: lm_train.py <config>""")
    parser.add_argument("config", help="path to config file")
    parser.add_argument('--continue-training', type=utils.str2bool, default=False,
                        help='Continue training from last_model.pt.')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    timer = utils.Timer()
    args = get_args()
    timer.tic()    
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)       
    dataconfig = config["data"]
    trainingconfig = config["training"]
    modelconfig = config["model"]

    training_set = data.TextLineByLineDataset(dataconfig["trainset"])
    valid_set = data.TextLineByLineDataset(dataconfig["devset"])
    if "vocab_path" in dataconfig:
        tokenizer = data.CharTokenizer(dataconfig["vocab_path"])
    else:
        raise ValueError("Unknown tokenizer.")

    
    modelconfig['vocab_size'] = tokenizer.unit_num() 
    collate = data.TextCollate(tokenizer, dataconfig["maxlen"])
 
    ngpu = 1 
    if "multi_gpu" in trainingconfig and trainingconfig["multi_gpu"] == True:
        ngpu = torch.cuda.device_count()
       
    tr_loader = torch.utils.data.DataLoader(training_set, 
        collate_fn=collate, batch_size=trainingconfig['batch_size'], shuffle=True, num_workers=dataconfig["fetchworker_num"])
    cv_loader = torch.utils.data.DataLoader(valid_set, 
        collate_fn=collate, batch_size=trainingconfig['batch_size'], shuffle=False, num_workers=dataconfig["fetchworker_num"])

    if modelconfig["type"] == "lstm":
        lmlayer = lm_layers.LSTM(modelconfig)
    else:
        raise ValueError("Unknown model")

    model = models.LM(lmlayer)
    logging.info("\nModel info:\n{}".format(model))   
    
    if args.continue_training:
        logging.info("Load package from {}.".format(os.path.join(trainingconfig["exp_dir"], "last-ckpt.pt")))
        pkg = torch.load(os.path.join(trainingconfig["exp_dir"], "last-ckpt.pt")) 
        model.restore(pkg["model"])        
 
    if "multi_gpu" in trainingconfig and trainingconfig["multi_gpu"] == True:
        logging.info("Let's use {} GPUs!".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)

    model = model.cuda()
    
    trainer = LmTrainer(model, trainingconfig, tr_loader, cv_loader)
    
    if args.continue_training:
        logging.info("Restore trainer states...")
        trainer.restore(pkg)
    logging.info("Start training...")
    trainer.train()
    logging.info("Total time: {:.4f} mins".format(timer.toc()/60.))

