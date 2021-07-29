import os
import sys
import time


def main(opt, mode=2):
    if opt == 'preprocess':
        command = "python preprocess.py -train_src samples/%s/train/train.spl.src \
                        -train_tgt samples/%s/train/train.txt.tgt \
                        -valid_src samples/%s/valid/valid.spl.src \
                        -valid_tgt samples/%s/valid/valid.txt.tgt \
                        -save_data samples/%s/preprocessed/baseline_spl \
                        -src_seq_length 10000 \
                        -tgt_seq_length 10000 \
                        -src_seq_length_trunc %d \
                        -tgt_seq_length_trunc %d" % (lang, lang, lang, lang, lang, src_len, tgt_len)
        os.system(command)
    elif opt == 'train':
        command = "python train.py -word_vec_size 256 \
                        -layers 1 \
                        -rnn_size 512 \
                        -rnn_type LSTM \
                        -global_attention mlp \
                        -data samples/%s/preprocessed/baseline_spl \
                        -save_model models/%s/baseline_spl \
                        -gpu_ranks 0 \
                        -batch_size 32 \
                        -optim adam \
                        -learning_rate 0.001 \
                        -dropout 0 \
                        -encoder_type brnn" % (lang, lang)
        os.system(command)
    elif opt == 'retrieval':
        print('Syntactic level...')
        command1 = "python syntax.py %s" % lang
        os.system(command1)
        print('Semantic level...')
        batch_size = 16 if lang == 'python' else 8
        command2 = "python translate.py -model models/%s/baseline_spl_step_100000.pt \
                        -src samples/%s/train/train.spl.src \
                        -output samples/%s/output/test.out \
                        -batch_size %d \
                        -gpu 0 \
                        -fast \
                        -max_sent_length %d \
                        -refer 0 \
                        -lang %s \
                        -search 2" % (lang, lang, lang, batch_size, src_len, lang)
        os.system(command2)
        command3 = "python translate.py -model models/%s/baseline_spl_step_100000.pt \
                        -src samples/%s/test/test.spl.src \
                        -output samples/%s/test/test.ref.src.1 \
                        -batch_size 16 \
                        -gpu 0 \
                        -fast \
                        -max_sent_length %d \
                        -refer 0 \
                        -lang %s \
                        -search 2" % (lang, lang, lang, src_len, lang)
        os.system(command3)
        print('Normalize...')
        command4 = "python normalize.py %s" % lang
        os.system(command4)
    elif opt == 'translate':
        command = "python translate.py -model models/%s/baseline_spl_step_100000.pt \
                    -src samples/%s/test/test.spl.src \
                    -output samples/%s/output/test.out \
                    -min_length 3 \
                    -max_length %d \
                    -batch_size 32 \
                    -gpu 0 \
                    -fast \
                    -max_sent_length %d \
                    -refer %d \
                    -lang %s \
                    -beam 5" % (lang, lang, lang, tgt_len, src_len, mode, lang)
        os.system(command)
        print('Done.')


if __name__ == '__main__':
    option = sys.argv[1]
    lang = sys.argv[2]
    assert option in ['preprocess', 'train', 'retrieval', 'translate', 'all']
    assert lang in ['python', 'java']
    if lang == 'python':
        src_len, tgt_len = 100, 50
    elif lang == 'java':
        src_len, tgt_len = 300, 30
    else:
        print("Unsupported Programming Language:", lang)
    if option == 'all':
        main('preprocess')
        main('train')
        main('retrieval')
        main('translate')
    else:
        if option == 'translate':
            mode = int(sys.argv[3])
            main(option, mode)
        else:
            main(option)
'''
Namespace(alpha=0.0, attn_debug=False, avg_raw_probs=False, batch_size=32, beam_size=5, 
beta=-0.0, block_ngram_repeat=0, coverage_penalty='none', data_type='text', dump_beam='', 
dynamic_dict=False, fast=True, gpu=0, guide=0, ignore_when_blocking=[], image_channel_size=3,
 lang='java', length_penalty='none', log_file='', log_file_level='0', lower=False, max_length=30,
  max_sent_length=300, min_length=3, models=['models/java/baseline_spl_step_100000.pt'], 
  n_best=1, output='samples/java/output/test.out', refer=2, replace_unk=False, report_bleu=False,
   report_rouge=False, sample_rate=16000, search=0, share_vocab=False, 
   src='samples/java/test/test.spl.src', src_dir='', stepwise_penalty=False, tgt=None,
 verbose=False, window='hamming', window_size=0.02, window_stride=0.01, zx='zx')
'''