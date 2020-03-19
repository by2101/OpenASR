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
import numpy as np
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


def get_args():
    parser = argparse.ArgumentParser(description="""
     Usage: stat_lengths.py <data.json> <vocab_file>""")
    parser.add_argument("data", help="path to data.")
    parser.add_argument("vocab_file", help="path to vocabulary file.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    vocab_path = args.vocab_file
    data_path = args.data
    training_set = data.SpeechDataset(data_path)
    if vocab_path.endswith(".model"):
        tokenizer = data.WpmTokenizer(vocab_path)
    else:
        tokenizer = data.CharTokenizer(vocab_path)
 
    durs = []
    id_lengths = []
    for d in iter(training_set):
        durs.append(d["duration"])
        ids = tokenizer.encode(d["transcript"]) 
        #print(ids)
        id_lengths.append(len(ids))
    durs = np.array(durs)
    id_lengths = np.array(id_lengths).astype(np.float)
    dur_percentile = np.percentile(durs, [10, 50, 90]) 
    dur_max = np.max(durs)
    dur_min = np.min(durs)
    dur_mean = np.mean(durs)
    msg = ("duration statistics:\n" +
            "max: {:.4f}s | min {:.4f}s | mean {:.4f}\n".format(dur_max, dur_min, dur_mean) +
            "percentile at (10, 50, 90): {}s {}s {}s\n".format(dur_percentile[0], dur_percentile[1], dur_percentile[2]))

    id_len_percentile = np.percentile(id_lengths, [10, 50, 90])
    id_len_max = np.max(id_lengths)
    id_len_min = np.min(id_lengths)
    id_len_mean = np.mean(id_lengths)
    msg += ("ids length statistics:\n" +
           "max: {:.4f} | min {:.4f} | mean {:.4f}\n".format(id_len_max, id_len_min, id_len_mean) +
           "percentile at (10, 50, 90): {} {} {}\n".format(id_len_percentile[0], id_len_percentile[1], id_len_percentile[2]))
    logging.info("\n"+msg)
