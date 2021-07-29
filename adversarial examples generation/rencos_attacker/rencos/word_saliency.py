import numpy as np
from replace_and_camelSplit import replace_a_to_b,split_c_and_s

from get_encoder import return_encoder
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
import time
import numpy as np
import sys

UNK='<unk>'



def computer_word_saliency_cos(model,code,summary,variable_list,vocab,embeddings,max_token,
                               vocab_src,vocab_trg,max_token_src,max_token_trg):
    word_saliency_list=[]
    code_=split_c_and_s(code.split(' '))
   
    encoder=return_encoder(code_,summary,model,vocab_src,vocab_trg,max_token_src,max_token_trg)

    for var in variable_list:
        var_index = vocab[var].index if var in vocab else max_token
        embedding_var=embeddings[var_index]

        cos=cosine_similarity(embedding_var.reshape(1,64),encoder.reshape(1,64))[0][0]
        cos=(1.0+cos)*0.5
        word_saliency_list.append((var,cos))
    return word_saliency_list


def return_bleu(dataiter,data,builder, model, summary,first,ori_data_token=''):
    translator=model
    
    batch_result = translator.translate_batch_adv(dataiter, data,first) #,attn_debug=False,ori_data_token='', fast=True)
    
    _,batch_sent = translator.transToSent(builder, batch_result)
    
    hyp=batch_sent[0][0].split(' ')
    ref=summary.split(' ')
    bleu_score = sentence_bleu([ref], hyp, smoothing_function = SmoothingFunction().method4)

    return bleu_score*100

def computer_best_substitution(model,code,summary,dataiter,data,builder,variable_list,nearest_k_dict):

    best_substitution_list=[]
    code_=split_c_and_s(code.split(' '))
   
    old_bleu=return_bleu(dataiter,data,builder, model, summary,True) 
      
    for var in variable_list:  # variable list  for one code
        max_delta_bleu=0
        nearest_k=nearest_k_dict[var]   # a list
        best_new_var =nearest_k[0]

        for new_var in nearest_k:
            new_code_list=replace_a_to_b(code,var,new_var)
            new_code=split_c_and_s(new_code_list)
            
            
            new_bleu=return_bleu(dataiter,data,builder, model, summary,False,ori_data_token=new_code)
            
            delta_bleu=old_bleu-new_bleu
            
            if max_delta_bleu< delta_bleu:
                max_delta_bleu=delta_bleu
                best_new_var=new_var

        best_substitution_list.append((var,best_new_var,max_delta_bleu))
    return best_substitution_list