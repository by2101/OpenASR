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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')

def get_args():
    parser = argparse.ArgumentParser(description="""
     Usage: avg_last_ckpts.py <expdir> <num>""")
    parser.add_argument("expdir", help="The directory contains the checkpoints.")
    parser.add_argument("num", type=int, help="The number of models to average")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    fnckpts = [t for t in os.listdir(args.expdir) if t.startswith("ep-") and t.endswith(".pt")]
    fnckpts.sort()
    fnckpts.reverse()
    fnckpts = fnckpts[:args.num]
    logging.info("Average checkpoints:\n{}".format("\n".join(fnckpts)))
    pkg = torch.load(os.path.join(args.expdir, fnckpts[0]), map_location=lambda storage, loc: storage)
    for k in pkg["model"]:
        if k.endswith("_state"):
            for key in pkg["model"][k].keys():
                pkg["model"][k][key] = torch.zeros_like(pkg["model"][k][key])

    for fn in fnckpts:
        pkg_tmp = torch.load(os.path.join(args.expdir, fn), map_location=lambda storage, loc: storage)
        logging.info("Loading {}".format(os.path.join(args.expdir, fn)))
        for k in pkg["model"]:
            if k.endswith("_state"):
                for key in pkg["model"][k].keys():
                    pkg["model"][k][key] += pkg_tmp["model"][k][key]/len(fnckpts)
    fn_save = os.path.join(args.expdir, "avg-last{}.pt".format(len(fnckpts)))
    logging.info("Save averaged model to {}.".format(fn_save))
    torch.save(pkg, fn_save)
       

    






