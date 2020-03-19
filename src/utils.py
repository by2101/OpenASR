import os
import io
import logging
import subprocess
import time
import math
import numpy as np
import torch
import torch.nn as nn
from third_party import wavfile
from third_party import kaldi_io as kio

TENSORBOARD_LOGGING = 1

def cleanup_ckpt(expdir, num_last_ckpt_keep):
    ckptlist = [t for t in os.listdir(expdir) if t.endswith('.pt') and t != 'last-ckpt.pt']
    ckptlist = sorted(ckptlist)
    ckptlist_rm = ckptlist[:-num_last_ckpt_keep]
    logging.info("Clean up checkpoints. Remain the last {} checkpoints.".format(num_last_ckpt_keep))
    for name in ckptlist_rm:
       os.remove(os.path.join(expdir, name))  


def get_command_stdout(command, require_zero_status=True):
    """ Executes a command and returns its stdout output as a string.  The
        command is executed with shell=True, so it may contain pipes and
        other shell constructs.

        If require_zero_stats is True, this function will raise an exception if
        the command has nonzero exit status.  If False, it just prints a warning
        if the exit status is nonzero.

        See also: execute_command, background_command
    """
    p = subprocess.Popen(command, shell=True,
                         stdout=subprocess.PIPE)

    stdout = p.communicate()[0]
    if p.returncode is not 0:
        output = "Command exited with status {0}: {1}".format(
            p.returncode, command)
        if require_zero_status:
            raise Exception(output)
        else:
            logger.warning(output)
    return stdout

def load_wave(path):
    """
    path can be wav filename or pipeline
    """

    # parse path
    items = path.strip().split(":", 1)
    if len(items) != 2:
        raise ValueError("Unknown path format.")
    tag = items[0]
    path = items[1]
    if tag == "file":
        sample_rate, data = wavfile.read(path)
    elif tag == "pipe":
        path = path[:-1]
        out = get_command_stdout(path, require_zero_status=True)
        sample_rate, data = wavfile.read(io.BytesIO(out))
    elif tag == "ark": 
        fn, offset = path.split(":", 1)
        offset = int(offset)
        with open(fn, 'rb') as f:
            f.seek(offset)
            sample_rate, data = wavfile.read(f, offset=offset)
    else:
        raise ValueError("Unknown file tag.")
    data = data.astype(np.float32)
    return sample_rate, data
    

def load_feat(path):
    items = path.strip().split(":", 1)
    if len(items) != 2:
        raise ValueError("Unknown path format.")
    tag = items[0]
    path = items[1]
    if tag == "ark": 
        return kio.read_mat(path)
    else:
        raise ValueError("Unknown file tag.")


def parse_scp(fn):
    dic = {}
    with open(fn, 'r') as f:
        cnt = 0
        for line in f:
            cnt += 1
            items = line.strip().split(' ', 1)
            if len(items) != 2:
                logging.warning('Wrong formated line {} in scp {}, skip it.'.format(cnt, fn))
                continue
            dic[items[0]] = items[1]
    return dic

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

class Timer(object):
    def __init__(self):
        self.start = 0.

    def tic(self):
        self.start = time.time()

    def toc(self):
        return time.time() - self.start



# ==========================================
# auxilary functions for sequence
# ==========================================

def get_paddings(src, lengths):
    paddings = torch.zeros_like(src).to(src.device)
    for b in range(lengths.shape[0]):
        paddings[b, lengths[b]:, :] = 1
    return paddings

def get_paddings_by_shape(shape, lengths, device="cpu"):
    paddings = torch.zeros(shape).to(device)
    if shape[0] != lengths.shape[0]:
        raise ValueError("shape[0] does not match lengths.shape[0]:"
            " {} vs. {}".format(shape[0], lengths.shape[0]))
    T = shape[1]
    for b in range(shape[0]):
        if lengths[b] < T:
            l = lengths[b]
            paddings[b, l:] = 1
    return paddings

def get_transformer_padding_byte_masks(B, T, lengths):
    masks = get_paddings_by_shape([B, T], lengths).byte()
    return masks

def get_transformer_casual_masks(T):
    masks = -torch.triu(
            torch.ones(T, T), diagonal=1)*9e20
    return masks


# ==========================================
# visualization
# ==========================================
if TENSORBOARD_LOGGING == 1:
    import logging
    mpl_logger = logging.getLogger("matplotlib")
    mpl_logger.setLevel(logging.WARNING)
    
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    from tensorboardX import SummaryWriter
    
    class Visualizer(object):
        def __init__(self):
            self.writer = None
            self.fig_step = 0
        
        def set_writer(self, log_dir):
            if self.writer is not None:
                raise ValueError("Dont set writer twice.")
            self.writer = SummaryWriter(log_dir)
                
        def add_scalar(self, tag, value, step):
            self.writer.add_scalar(tag=tag, 
                scalar_value=value, global_step=step)
                
        def add_graph(self, model):
            self.writer.add_graph(model)

        def add_image(self, tag, img, data_formats):
            self.writer.add_image(tag, 
                img, 0, dataformats=data_formats)
                
        def add_img_figure(self, tag, img, step=None):
            fig, axes = plt.subplots(1,1)
            axes.imshow(img)
            self.writer.add_figure(tag, fig, global_step=step)                
                
        def close(self):
            self.writer.close()
            
    visualizer = Visualizer()            

