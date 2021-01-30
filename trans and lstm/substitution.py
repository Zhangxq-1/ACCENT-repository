from word_saliency import computer_best_substitution,computer_word_saliency_cos
from main.model import Code2NaturalLanguage   #main   return encoder vector
from gensim.models.word2vec import Word2Vec
import torch
torch.cuda.current_device()
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import time
import numpy as np
from encoder.rnnModel import Seq2Seq

def softmax(x):
    exp_x=np.exp(x)
    softmax_x=exp_x/np.sum(exp_x)
    return softmax_x

def get_topk_index(k,arr):
    top_k=k
    array=arr
    top_k_index=array.argsort()[::-1][0:top_k]
    return top_k_index

def rank_variable(code,summary,variable_list,nearest_k_dict,vocab,embeddings,max_token,
                  model_encoder,vocab_src,vocab_trg,max_token_src,max_token_trg):




    word_saliency_list = computer_word_saliency_cos(model_encoder, code, summary, variable_list, vocab,embeddings,max_token,
                                                    vocab_src, vocab_trg, max_token_src, max_token_trg
                                                    )
    #print(word_saliency_list)
    best_substitution_list=computer_best_substitution(model,code,summary,variable_list,nearest_k_dict)
    #print(best_substitution_list)
    unk_delta_bleu=[]
    best_delta_bleu=[]
    best_sub_list=[]
    for item in word_saliency_list:
        unk_delta_bleu.append(item[1])
    for item in best_substitution_list:
        best_delta_bleu.append(item[2])
        best_sub_list.append(item[1])

    np_unk_delta_bleu=np.array(unk_delta_bleu)
    np_best_delta_bleu=np.array(best_delta_bleu)
    np_unk_delta_bleu_soft=softmax(np_unk_delta_bleu)
    sorce=np_unk_delta_bleu_soft*np_best_delta_bleu
    
    for i in range(len(sorce)):
        if sorce[i]==0 and np_unk_delta_bleu_soft[i]!=0:
            sorce[i]=np_unk_delta_bleu_soft[i]*0.5 
        if sorce[i]==0 and np_best_delta_bleu[i]!=0:
            sorce[i]=np_best_delta_bleu[i]
    
    print(sorce)
    descend_index=get_topk_index(len(sorce),sorce)
    print(descend_index)
    descend_variable={}
    for item in descend_index:
        var=variable_list[item]
        sub=best_sub_list[item]
        descend_variable[var]=sub
    print(descend_variable)
    return descend_variable

if __name__=='__main__':
    print('start at time:')
    print(time.strftime(' %H:%M:%S',time.localtime(time.time())))
    #unk='<unk>'
    data_root = ' '
    f_code = open('/data/python/test/code.original', 'r')
    f_summary = open('/data/python/test/javadoc.original', 'r')
    nearest_k_data = pd.read_pickle(
        '/data/python/nearest_k_for_everyVar_test.pkl')
    var_everyCode_data = pd.read_pickle(
        'data/python/var_for_everyCode_test.pkl')

    nearest_k_list = nearest_k_data['nearest_k'].tolist()
    var_everyCode_list = var_everyCode_data['variable'].tolist()
    model_file=' '   ###original trained code comment generation model file
    
    model = Code2NaturalLanguage.load(model_file)

    print('read embedding....')
    root = ' '
    word2vec = Word2Vec.load('/embedding/python/node_w2v_64').wv
    vocab = word2vec.vocab
    max_token = word2vec.syn0.shape[0]
    embedding_dim = word2vec.syn0.shape[1]
    embeddings = np.zeros((max_token + 1, embedding_dim))
    embeddings[:max_token] = word2vec.syn0
    print('end read embedding..')

    print('read embedding of src and trg....')
    word2vec_src = Word2Vec.load(
        '/encoder/vocab/python/python_node_w2v_code_64').wv
    vocab_src = word2vec_src.vocab
    max_token_src = word2vec_src.syn0.shape[0]
    embedding_dim = word2vec_src.syn0.shape[1]
    embeddings_src = np.zeros((max_token_src + 1, embedding_dim))
    embeddings_src[:max_token_src] = word2vec_src.syn0

    word2vec_trg = Word2Vec.load(
        '/encoder/vocab/python/python_node_w2v_summ_64').wv
    vocab_trg = word2vec_trg.vocab
    max_token_trg = word2vec_trg.syn0.shape[0]
    embedding_dim = word2vec_trg.syn0.shape[1]
    embeddings_trg = np.zeros((max_token_trg + 1, embedding_dim))
    embeddings_trg[:max_token_trg] = word2vec_trg.syn0

    print('load encoder model:')
    model_encoder = Seq2Seq(embeddings_src,embeddings_trg)
    model_encoder.load_state_dict(torch.load('/encoder/model/python_model.pkl'))



    #---------------------------------------------
    best_descend_var_list=[]
    count=0
    save=0
    for code,summ in zip(f_code,f_summary):


        variable_list=var_everyCode_list[count]
        nearest_k_dict=nearest_k_list[count]
        descend_variable_dict=rank_variable(code,summ,variable_list,nearest_k_dict,vocab,embeddings,max_token,
                                            model_encoder,vocab_src,vocab_trg,max_token_src,max_token_trg)
        best_descend_var_list.append(descend_variable_dict)
        count=count+1
        print('ok'+str(count))
        '''
        
        if count%5000==0:
            index = [i for i in range(len(best_descend_var_list))]
            data_descend_best_var = pd.DataFrame({'id': index, 'var_sub': best_descend_var_list})
            data_descend_best_var.to_pickle(
                '/home/zxq/code/attacker_transformer/attacker/var_name/python/train/trans/train_trans_data_descend_best_var_8.pkl')
            save=save+1
            print('saved')
        '''
        
    index=[i for i in range(len(best_descend_var_list))]
    data_descend_best_var=pd.DataFrame({'id':index,'var_sub':best_descend_var_list})
    data_descend_best_var.to_pickle(
        '/data/python/test_rnn_data_descend_best_var.pkl')
    print('end...')


