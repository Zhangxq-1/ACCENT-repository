#!/usr/bin/env bash

function make_dir () {
    if [[ ! -d "$1" ]]; then
        mkdir $1
    fi
}

SRC_DIR=../..
DATA_DIR=${SRC_DIR}/data
MODEL_DIR=${SRC_DIR}/tmp-adv-rnn-python

make_dir $MODEL_DIR

DATASET=python
CODE_EXTENSION=original_subtoken
JAVADOC_EXTENSION=original


function train () {

echo "============TRAINING============"

RGPU=$1
MODEL_NAME=$2

PYTHONPATH=$SRC_DIR CUDA_VISIBLE_DEVICES=$RGPU python -W ignore ${SRC_DIR}/main/train.py \
--data_workers 5 \
--dataset_name $DATASET \
--data_dir ${DATA_DIR}/ \
--model_dir $MODEL_DIR \
--model_name $MODEL_NAME \
--train_src train/code.${CODE_EXTENSION} \
--train_src_tag train/code.${CODE_EXTENSION} \
--train_tgt train/javadoc.${JAVADOC_EXTENSION} \
--dev_src dev/code.${CODE_EXTENSION} \
--dev_src_tag dev/code.${CODE_EXTENSION} \
--dev_tgt dev/javadoc.${JAVADOC_EXTENSION} \
--code_tag_type $CODE_EXTENSION \
--use_code_type False \
--uncase True \
--use_src_word True \
--use_src_char False \
--use_tgt_word True \
--use_tgt_char False \
--max_src_len 400 \
--max_tgt_len 30 \
--emsize 512 \
--fix_embeddings False \
--src_vocab_size 50000 \
--tgt_vocab_size 30000 \
--share_decoder_embeddings True \
--conditional_decoding False \
--max_examples -1 \
--batch_size 32 \
--test_batch_size 64 \
--num_epochs 200 \
--model_type rnn \
--nhid 512 \
--nlayers 2 \
--dropout_rnn 0.2 \
--dropout_emb 0.2 \
--dropout 0.2 \
--copy_attn True \
--reuse_copy_attn True \
--early_stop 20 \
--optimizer adam \
--learning_rate 0.002 \
--lr_decay 0.99 \
--grad_clipping 5.0 \
--valid_metric bleu \
--checkpoint True

}

function test () {

echo "============TESTING============"

RGPU=$1
MODEL_NAME=$2

PYTHONPATH=$SRC_DIR CUDA_VISIBLE_DEVICES=$RGPU python -W ignore ${SRC_DIR}/main/train.py \
--only_test True \
--data_workers 5 \
--dataset_name $DATASET \
--data_dir ${DATA_DIR}/ \
--model_dir $MODEL_DIR \
--model_name $MODEL_NAME \
--dev_src test/code.${CODE_EXTENSION} \
--dev_src_tag test/code.${CODE_EXTENSION} \
--dev_tgt test/javadoc.${JAVADOC_EXTENSION} \
--code_tag_type $CODE_EXTENSION \
--use_code_type False \
--uncase True \
--max_src_len 400 \
--max_tgt_len 30 \
--max_examples -1 \
--test_batch_size 64

}

#gnncode_adv_3_new
#gnn_MH_adv_code_mh2_3_new

#cscgcode_adv_3       now
#CSCG_MH2_adv_code_3
#code_random_adv2
#code_random_adv3


function beam_search () {

echo "============Beam Search TESTING============"

RGPU=$1
MODEL_NAME=$2

PYTHONPATH=$SRC_DIR CUDA_VISIBLE_DEVICES=$RGPU python -W ignore ${SRC_DIR}/main/test.py \
--data_workers 5 \
--dataset_name $DATASET \
--data_dir ${DATA_DIR}/ \
--model_dir $MODEL_DIR \
--model_name $MODEL_NAME \
--dev_src test/rnn_adv_code_3.${CODE_EXTENSION} \
--dev_src_tag test/rnn_adv_code_3.${CODE_EXTENSION} \
--dev_tgt test/javadoc.${JAVADOC_EXTENSION} \
--code_tag_type $CODE_EXTENSION \
--use_code_type False \
--uncase True \
--max_examples -1 \
--max_src_len 400 \
--max_tgt_len 30 \
--test_batch_size 64 \
--beam_size 4 \
--n_best 1 \
--block_ngram_repeat 3 \
--stepwise_penalty False \
--coverage_penalty none \
--length_penalty none \
--beta 0 \
--gamma 0 \
--replace_unk

}

#train $1 $2
#test $1 $2
beam_search $1 $2
