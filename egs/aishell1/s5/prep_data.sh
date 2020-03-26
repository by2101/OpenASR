#!/bin/bash

source path.sh

corpusdir=/data1/Corpora/aishell
url=www.openslr.org/resources/33 

echo "============================================================================"
echo "Step 1: Download AISHELL1 dataset, and prepare KALDI style data directories."
echo "============================================================================"
bash local/download_and_untar.sh $corpusdir $url data_aishell
bash local/aishell_data_prep.sh $corpusdir/data_aishell/wav $corpusdir/data_aishell/transcript

echo "============================================================================"
echo "Step 2: Format data to json file for training Seq2Seq models."
echo "============================================================================"
echo "Remove space in transcripts"
for x in train dev test; do
    if [ ! -f $x/text.org ]; then
        cp data/$x/text data/$x/text.org
    fi
    cat data/$x/text.org | awk '{printf($1" "); for(i=2;i<=NF;i++){printf($i)}; printf("\n")}' > data/$x/text
done
echo "Prepare json format data files."
mkdir -p exp
for x in train dev test; do
    python $MAIN_ROOT/src/prepare_data.py --tag file data/$x exp/${x}.json
done
echo "Generate vocabulary"
python $MAIN_ROOT/src/stat_grapheme.py data/train/text exp/aishell1_train_chars.txt


