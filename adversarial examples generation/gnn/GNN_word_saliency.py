import numpy as np
from replace_and_camelSplit import replace_a_to_b,split_c_and_s,replace_a_to_b_2
from GNN_get_bleu import return_bleu
from sklearn.metrics.pairwise import cosine_similarity

from  get_encoder import return_encoder

# return a list for every code (sample) [(variable,bleu_old-bleu_new),(variable,bleu_old-bleu_new)..]
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


def computer_best_substitution(test,code,ast,edg,summary,variable_list,nearest_k_dict):

    best_substitution_list=[]
    code_=split_c_and_s(code.split(' '))
    old_bleu =return_bleu(code_,ast,edg,summary,test)
    old_bleu=old_bleu*100.0
    EDG=edg
    for var in variable_list:
        max_delta_bleu=0
        nearest_k=nearest_k_dict[var]
        best_new_var =nearest_k[0]

        for new_var in nearest_k:
            new_code_list=replace_a_to_b(code,var,new_var)
            new_code=split_c_and_s(new_code_list)

            new_ast=replace_a_to_b_2(ast,var,new_var)

            new_bleu =return_bleu(new_code,new_ast,EDG,summary,test)
            new_bleu=new_bleu*100.0
      
            delta_bleu=old_bleu-new_bleu
            if max_delta_bleu< delta_bleu:
                max_delta_bleu=delta_bleu
                best_new_var=new_var

        best_substitution_list.append((var,best_new_var,max_delta_bleu))
    return best_substitution_list