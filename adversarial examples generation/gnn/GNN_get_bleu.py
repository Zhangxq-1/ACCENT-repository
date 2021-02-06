
import os
import torch
import torch.nn as nn

from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction


def maxpooling(x):
    x=torch.max(x,2)[0]
    x=x.view(1,1,x.shape[1])
    out=nn.MaxPool1d(4,stride=4)
    x=out(x)
    x = x.squeeze()
    return x  #[64]


def tensor_to_numpy(x):
    return x.cpu().numpy()



def com_bleu(ref,gene):

    ref_=[ref.split(' ')]
    #gene_=gene.split(' ')
    gene_=gene[0]
    bleu=sentence_bleu(ref_, gene_, smoothing_function=SmoothingFunction().method4)
    return bleu


def return_bleu(code, ast, edg, summary, test):

    edg_batch=[edg]
    test.init_dataloader(ast, code, summary)
    candidates, _=test.test_for_adv(edg_batch)
    bleu=com_bleu(summary,candidates)

    return bleu

