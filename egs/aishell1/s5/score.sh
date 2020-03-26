#!/bin/bash

source path.sh

ref=$1
dir=$2

cat $ref | python3 -c \
"
import sys
for line in sys.stdin:
  utt,txt = line.strip().split(' ', 1)
  txt = ' '.join(list(txt))
  print('{} ({})'.format(txt, utt))
" > ${dir}/ref.trn

$MAIN_ROOT/tools/sclite -r ${dir}/ref.trn trn -h ${dir}/hyp.trn trn -c NOASCII -i wsj -o all stdout > ${dir}/result.txt

echo "write a CER (or TER) result in ${dir}/result.txt"
grep -e Avg -m 2 ${dir}/result.txt


