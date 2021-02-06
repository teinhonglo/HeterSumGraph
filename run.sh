#!/usr/bin/env bash

dataset="CNNDM"
data_dir="datasets/cnndm"
exp_dir="exp/cnndm"
gpu=0
n_epochs=20
stage=1

set -euo pipefail

if [ $stage -le 0 ]; then
    ./PrepareDataset.sh $dataset $data_dir || exit 0;
fi

if [ $stage -le 1 ]; then
    python train.py --cuda --gpu $gpu --data_dir $data_dir --n_epochs $n_epochs \
                --cache_dir cache/$dataset --embedding_path glove/glove.42B.300d.txt \
                --model HSG --save_root $exp_dir --log_root $exp_dir/log \
                --lr_descent --grad_clip -m 3 || exit 0;
fi

if [ $stage -le 2 ]; then
    python evaluation.py --cuda --gpu $gpu --data_dir $data_dir \
                         --cache_dir cache/$dataset \
                         --embedding_path glove/glove.42B.300d.txt \
                         --model HSG --save_root $exp_dir --log_root $exp_dir/log -m 3 --test_model multi --use_pyrouge || exit 0;
fi
