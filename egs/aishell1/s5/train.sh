#!/bin/bash

source path.sh
export CUDA_VISIBLE_DEVICES=0

sys_tag="base"
if [ $# != 0 ]; then
    sys_tag=$1
fi


if [ "$sys_tag" == "base" ]; then

echo "Training a baseline transformer ASR system..."
python $MAIN_ROOT/src/train.py config_base.yaml 2>&1 | tee base.log 

elif [ "$sys_tag" == "lm" ]; then
    cat data/train/text | cut -d" " -f2- > exp/train_text
    cat data/dev/text | cut -d" " -f2- > exp/dev_text
    python $MAIN_ROOT/src/lm_train.py config_lm_lstm.yaml 2>&1 | tee base.log 

elif [ "$sys_tag" == "lst" ]; then
    echo ""
else

    echo "The sys_tag should be base, lm or lst."
    exit 1
fi
