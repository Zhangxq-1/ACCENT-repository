import pandas as pd
import numpy as np
from gensim.models.word2vec import Word2Vec
import random
import os
from sklearn.metrics.pairwise import cosine_similarity

k=8
def get_topk_index(k,arr):
    top_k=k
    array=arr
    top_k_index=array.argsort()[::-1][0:top_k]
    return top_k_index   #return top-k use list[]


print('read embedding....')
root = ''
word2vec = Word2Vec.load('/embedding/python/node_w2v_64').wv
vocab=word2vec.vocab
max_token=word2vec.syn0.shape[0]
embedding_dim=word2vec.syn0.shape[1]
embeddings = np.zeros((max_token+1,embedding_dim))
embeddings[:max_token]=word2vec.syn0
print('end read embedding..')

print('read all var and extract embedding..')
data_all_var=pd.read_pickle('data/python/var_for_allCode_train.pkl')  # not change
all_var_list=list(data_all_var['all vars'].tolist()[0])
max_var=len(all_var_list)
embeddings_allvar=np.zeros((max_var,embedding_dim))
index_all_var=[]
for item in all_var_list:
    if item in vocab:
        index_all_var.append(vocab[item].index)
    else:
        index_all_var.append(max_token)

for i in range(max_var):
    embeddings_allvar[i]=embeddings[index_all_var[i]]
print('read and extract end')

print('read every var and formalparameter for every code')
data_every_var=pd.read_pickle('data/python/var_for_everyCode_test.pkl')
every_var_list=data_every_var['variable'].tolist()

#formalParameter
formalParameter_for_every_code=pd.read_pickle\
    ('data/python/formalParameter_for_everyCode_test.pkl')
formalParameter_list=formalParameter_for_every_code['variable'].tolist()

#select top k var expect for self code

print('select top k nearest var')
nearest_list=[]
var_embed=np.zeros((1,embedding_dim))
count=0

print('random select k var')
count=0
for every_code in every_var_list:  #for every code
    nearest_dict={}
    mask_index_list=[]
    formalParameter_every_code =formalParameter_list[count]

    for var in every_code:
        if var in all_var_list:
            var_index_in_all_var=all_var_list.index(var)
            mask_index_list.append(var_index_in_all_var)

    #jia shang dui formalParameter de chu li
    if formalParameter_every_code!=[]:
        for item in formalParameter_every_code:
            if item in all_var_list:
                mask_index_list.append(all_var_list.index(item))

    for var in every_code: #for every var of every code
        n_list = []
        i=0
        while i<k:
            random_index = random.sample(range(0, max_var), 1)
            if random_index[0] in mask_index_list:
                continue
            else:
                n_list.append(all_var_list[random_index[0]])
                i=i+1
        nearest_dict[var]=n_list
    nearest_list.append(nearest_dict)
    count=count+1
    print('ok'+str(count))


index=[i for i in range(len(nearest_list))]
nearest_pd=pd.DataFrame({'id':index,'nearest_k':nearest_list})
nearest_pd.to_pickle('/data/python/nearest_k_for_everyVar_test.pkl')
print('end')

