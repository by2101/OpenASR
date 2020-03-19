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
import argparse
import json
import os
import utils

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')


def get_args():
    parser = argparse.ArgumentParser(description="""
     Usage: prepare_data.py <data_dir> <dest_path>""")
    parser.add_argument("data_dir", help="data directory")
    parser.add_argument("dest_path", help="path to dest")
    parser.add_argument("--tag", type=str, default="file", 
            help="tag of path. It should be file, pipe, or ark.")
    parser.add_argument("--maxdur", type=float, default=9e9, 
            help="if the duration is longer than maxdur, drop it.")
    args = parser.parse_args()
    return args


def get_dur(wav_dic):
    durdic = {}
    for key, path in wav_dic.items():
        sample_rate, data = utils.load_wave(path)
        dur = data.shape[0]/float(sample_rate)
        durdic[key] = dur
    return durdic
    

if __name__ == "__main__":
    args = get_args()
    datadir = args.data_dir
    fw = args.dest_path
    logging.info("Preparing data for {}...".format(datadir)) 
    if os.path.exists(os.path.join(datadir, "wav.scp")):
        logging.info("wav.scp exists. Use it.")
        wav_dic = utils.parse_scp(os.path.join(datadir, "wav.scp"))
    elif os.path.exists(os.path.join(datadir, "feats.scp")):
        logging.info("wav.scp does not exists. Use feats.scp.")
        wav_dic = utils.parse_scp(os.path.join(datadir, "feats.scp"))
    else:
        raise ValueError("No speech scp.")
    trans_dic = utils.parse_scp(os.path.join(datadir, "text"))
    utts = wav_dic.keys()
    for utt in utts:
        wav_dic[utt] = "{}:{}".format(args.tag, wav_dic[utt])
    if os.path.exists(os.path.join(datadir, "utt2dur")):
        dur_dic = utils.parse_scp(os.path.join(datadir, "utt2dur"))
    else:
        logging.info("No utt2dur file, generate it.")
        dur_dic = get_dur(wav_dic)

    n_tot = 0
    n_success = 0
    n_durskip = 0
    towrite = []
    for utt in utts:
        n_tot += 1
        if utt not in trans_dic:
            logging.warn("No trans for {}, skip it.".format(utt))
            continue
        elif utt not in dur_dic:
            logging.warn("No dur for {}, skip it.".format(utt))
            continue

        if float(dur_dic[utt]) > args.maxdur:
            logging.warn("{} is longer than {}s, skip it.".format(utt, dur_dic[utt]))
            n_durskip += 1
            continue
        else: 
            towrite.append({
                "utt": utt,
                "path": wav_dic[utt],
                "transcript": trans_dic[utt],
                "duration": float(dur_dic[utt]),
                })
            n_success += 1
    with open(fw, 'w', encoding="utf8") as f:
        json.dump(towrite, f, ensure_ascii=False, indent=2)
    logging.info("\nProcessed {} utterances successfully. "
            "The total number is {}. ({:2%}) {} utterances are too long.".format(n_success, n_tot, 1.*n_success/n_tot, n_durskip))









