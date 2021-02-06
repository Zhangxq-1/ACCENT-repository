import pandas as pd
import torch
import numpy as np
import random

from nmt.replace_and_camelSplit import split_c_and_s

SUBSTITUTION=3
def generate_for_c2nl(ori_sents,var_all_list):
    unk_code_subtoken_list=[]
    for i in range(len(ori_sents)):
            code_list=ori_sents[i]
           
            
            var_list=var_all_list[i]

            sub_num=len(var_list) if len(var_list)<SUBSTITUTION else SUBSTITUTION
            masked_var_index=random.sample(range(0,len(var_list)),sub_num)
            old_list=[]
            for item in masked_var_index:
                old_list.append(var_list[item])


          
            new_code_list=[]
            for token in code_list:
                if token in old_list:
                    new_code_list.append('<unk>')
                else:
                    new_code_list.append(token)

            sub_code_new=split_c_and_s(new_code_list)     #STR
         
            unk_code_subtoken_list.append(sub_code_new.strip().split(' '))
  
    
    return unk_code_subtoken_list

def generate_for_nl2c(ori_sents):
    unk_code_subtoken_list=[]
    for i in range(len(ori_sents)):
            code_list=ori_sents[i]
           
            masked_var_index=random.sample(range(0,len(code_list)),SUBSTITUTION)
            old_list=[]
            for item in masked_var_index:
                old_list.append(code_list[item])


            
            new_code_list=[]
            for token in code_list:
                if token in old_list:
                    new_code_list.append('<unk>')
                else:
                    new_code_list.append(token)
            unk_code_subtoken_list.append(new_code_list)
    
    return unk_code_subtoken_list

            


            



