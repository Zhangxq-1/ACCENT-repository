from gensim.models.word2vec import Word2Vec
import pandas as pd
import os


def train_vocab(size):
    corpus=[]
    f_code=open('/data/java/train/code.original_subtoken','r')
    print('read original code')
    for line in f_code:
        code=line.split(' ')
        corpus.append(code)
    f_code.close()
    print('read end')

    if not os.path.exists( '/encoder/vocab/train'):
        os.mkdir('/encoder/vocab/train')

    w2v = Word2Vec(corpus, size=size, workers=16, sg=1, min_count=3)
    w2v.save('/encoder/vocab/train/node_w2v_code_' + str(size))
    MAX_TOKENS = w2v.wv.syn0.shape[0]
    print('max:'+str(MAX_TOKENS))

if __name__=='__main__':
    print('start training word2vec:')
    train_vocab(64)
    print('training end!')