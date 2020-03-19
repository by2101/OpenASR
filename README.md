# OpenASR

A pytorch based end2end speech recognition system. The main architecture is Speech-Transformer.

## Features

1. **Minimal Dependency**. The system does not depend on external softwares for feature extraction or decoding. Users just install PyTorch deep learning framework.
2. **Good Performance**. The system includes advanced algorithms, such as Label Smoothing, SpecAug, LST, and achived good performance on ASHELL1. The baseline CER on AISHELL1 test is 7.1.
3. **Modular Design**. We divided the system into several modules, such as trainer, metric, schedule, models. It is easy for extension and adding features.
4. **End2End**. The feature extraction and tokenization are online. So the system directly processes wave file.

## Usage
### Data Preparation




## Results




## Acknowledgement
This system is implemented with PyTorch. We use wave reading codes from SciPy. Thanks to Dan Povey's team and their KALDI software. I learn ASR concept, and example organization from KALDI. And thanks to Google Lingvo Team. I learn the modular design from Lingvo.

## 
@article{bai2019learn,
  title={Learn Spelling from Teachers: Transferring Knowledge from Language Models to Sequence-to-Sequence Speech Recognition},
  author={Bai, Ye and Yi, Jiangyan and Tao, Jianhua and Tian, Zhengkun and Wen, Zhengqi},
  year={2019}
}

## References
