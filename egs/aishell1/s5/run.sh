#!/bin/bash


bash prep_data.sh
bash train.sh
bash decode_test.sh
bash avg.sh
bash score.sh data/test/text exp/exp1/decode_test_avg-last10




