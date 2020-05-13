# OpenASR

基于Pytorch的端到端语音识别系统. 主要结构使用 [Speech-Transformer](https://ieeexplore.ieee.org/abstract/document/8462506/).

[README](https://github.com/by2101/OpenASR/blob/master/README.md)

## 主要特性

1. **最小依赖**. 系统不依赖其它额外的软件来提取特征或是解码。 用户只需要安装Pytorch即可。
2. **性能优良**. 系统集成了多个算法，包括Label Smoothing, SpecAugmentation, LST 等。在AISHELL-1数据集上，基线系统CER为6.6%，好于ESPNet。
3. **模块化设计**. 系统分为trainer, metric, schedule等模块，方便进一步扩展。
4. **端到端实现**. 特征提取和token划分采用在线实现。系统可以直接处理wav文件，整个流程大大简化。

## 依赖
python >= 3.6
pytorch >= 1.1
pyyaml >= 5.1
tensorflow 和 tensorboardX (如果不需要可视化，可以将src/utils.py中TENSORBOARD_LOGGING变量设为0)

## 使用方法
我们采用KALDI风格的例子。例子的目录包括一些高层脚本，data目录和exp目录。我们提供了一个AISHELL-1的例子，位于ROOT/egs/aishell1/s5.

### 数据准备
数据准备的脚本是prep_data.sh。它会自动地下载AISHELL-1数据集，并将数据整理成KALDI风格的data目录。然后，它会生成json数据文件和字表。你可以设置`corpusdir` 来改变存储数据的目录。

    bash prep_data.sh


### 训练模型
我们采用yaml文件来配置参数。我们提供3个例子。

    config_base.yaml  # 基线 ASR 
    config_lm_lstm.yaml  # LSTM 语言模型
    config_lst.yaml  # 采用LST训练的ASR

运行 train.sh 脚本训练基线系统。

    bash train.sh
    
### 模型平均
我们采用模型平均来提高性能。

    bash avg.sh
    
### 解码和打分
运行 decode_test.sh 解码测试集。然后运行score.sh计算CER。

    bash decode_test.sh
    bash score.sh data/test/text exp/exp1/decode_test_avg-last10

## 可视化
我们提供基于TensorbordX的可视化。event文件保存在$expdir/log。你可以通过tensorboard来观察训练过程。

    tensorboard --logdir=$expdir --bind_all
    
然后你就可以在浏览器中观察 (http://localhost:6006).

例子:
![per token loss in batch](https://github.com/by2101/OpenASR/raw/master/figs/loss.png)
![encoder attention](https://github.com/by2101/OpenASR/raw/master/figs/enc_att.png)
![encoder-decoder attention](https://github.com/by2101/OpenASR/raw/master/figs/dec_enc_att.png)


## 致谢
系统是基于PyTorch实现的。我们采用了SciPy里的读取wav文件的代码。我们使用了SCTK来计算CER。感谢Dan Povey团队和他们的KALDI，ASR的概念，例子的组织是从KALDI里学到的。感谢Google的Lingvo团队，模块化设计从Lingvo里学了很多。

## Bib
@article{bai2019learn,
  title={Learn Spelling from Teachers: Transferring Knowledge from Language Models to Sequence-to-Sequence Speech Recognition},
  author={Bai, Ye and Yi, Jiangyan and Tao, Jianhua and Tian, Zhengkun and Wen, Zhengqi},
  year={2019}
}

## 引用
Dong, Linhao, Shuang Xu, and Bo Xu. "Speech-transformer: a no-recurrence sequence-to-sequence model for speech recognition." 2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2018.
Zhou, Shiyu, et al. "Syllable-based sequence-to-sequence speech recognition with the transformer in mandarin chinese." arXiv preprint arXiv:1804.10752 (2018).
