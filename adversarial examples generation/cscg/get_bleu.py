
import os
import torch
import torch.nn as nn
from nltk.translate.bleu_score import  sentence_bleu, SmoothingFunction

def read_corpus(line, source):
    data = []
  
    sent = line.strip().split(' ')
        # only append <s> and </s> to the target sentence
    if source == 'tgt':
        sent = ['<s>'] + sent + ['</s>']
    data.append(sent)
    return data

def com_bleu(references, hypotheses, state):
    bleu_score = 0.0

    for ref, hyp in zip(references, hypotheses):
 
        if len(hyp[1:-1]) == 1:
            continue
        bleu_score += sentence_bleu([ref[1:-1]], hyp[1:-1], smoothing_function = SmoothingFunction().method4)
 
    return bleu_score / len(hypotheses)

def decode(model, data, verbose=False):
    """
    decode the dataset and compute sentence level acc. and BLEU.
    """
    hypotheses = []

    data = list(data)
    # from multiprocessing import Pool
    # from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
    if type(data[0]) is tuple:
        for src_sent, tgt_sent in data:
            hyps = model.translate(src_sent)
     
            hypotheses.append(hyps)    
    else:
        for src_sent in data:
            hyps = model.translate(src_sent)
            hypotheses.append(hyps)

    return hypotheses


def return_bleu(model,code,summary):
    test_data_src = read_corpus(code, source='src')
    test_data_tgt = read_corpus(summary, source='tgt')
    test_data = zip(test_data_src, test_data_tgt)    

    hypotheses = decode(model, test_data, verbose=False)
    top_hypotheses = [hyps[0] for hyps in hypotheses]

    bleu_score = com_bleu([tgt for tgt in test_data_tgt], top_hypotheses, 'test')

    return bleu_score




