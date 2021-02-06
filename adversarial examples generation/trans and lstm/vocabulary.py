from gensim.models.word2vec import Word2Vec
import pandas as pd
import os


voc_root=''

def train_vocab(size):
    corpus=[]
    f_code=open('/data/python/train/code.original','r')
    print('read original code')
    for line in f_code:
        code=line.split(' ')
        corpus.append(code)
    f_code.close()
    print('read end')
    '''
    if not os.path.exists('/embedding/python'):
        os.mkdir( '/embedding/python')
    '''


    w2v = Word2Vec(corpus, size=size, workers=16, sg=1, min_count=1)
    w2v.save('/embedding/python/node_w2v_64')
    MAX_TOKENS = w2v.wv.syn0.shape[0]
    print('max:'+str(MAX_TOKENS))

if __name__=='__main__':
    print('start training word2vec:')
    train_vocab(64)
    print('training end!')