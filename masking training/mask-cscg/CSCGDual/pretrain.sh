#!/bin/sh

L1='code'
L2='nl'
JOB='pretrain'

data_dir="/CSCGDual/data/java"
vocab_bin="$data_dir/voc_sub.bin"
train_src="$data_dir/train.token.${L1}"
train_tgt="$data_dir/train.token.${L2}"
train_ori="$data_dir/code.original"


test_src="$data_dir/dev.token.${L1}"
test_tgt="$data_dir/dev.token.${L2}"
test_ori="$data_dir/code.original"
job_name="$JOB"
model_name="pretrain_models/c2nl-java.${job_name}"

python3 /home/zxq/code/adversarial_CSCG/CSCGDual/nmt/nmt.py \
    --cuda \
    --gpu 0 \
    --mode train \
    --vocab ${vocab_bin} \
    --save_to ${model_name} \
    --log_every 1500 \
    --valid_niter 7500 \
    --valid_metric bleu \
    --save_model_after 1 \
    --beam_size 5 \
    --batch_size 8 \
    --hidden_size 512 \
    --embed_size 512 \
    --uniform_init 0.1 \
    --dropout 0.2 \
    --clip_grad 5.0 \
    --decode_max_time_step 50 \
    --lr_decay 0.9 \
    --lr 0.002 \
    --process c2nl \
    --train_src ${train_src} \
    --train_tgt ${train_tgt} \
    --train_ori ${train_ori} \
    --dev_src ${test_src} \
    --dev_tgt ${test_tgt} \

