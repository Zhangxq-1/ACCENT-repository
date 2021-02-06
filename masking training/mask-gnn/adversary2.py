import pandas as pd

import torch
import numpy as np
from replace_and_camelSplit import split_c_and_s2
import random
def load_pkl(path):
    return pd.read_pickle(path)

def get_original(original_path):
    f=open(original_path)
    l=[]
    for line in f:
        l.append(line)   
    return l
    
class Adversary(object):
    def __init__(self,var_everyCode_path,original_code_path):
        
        self.var_everyCode_list = load_pkl(var_everyCode_path)['variable'].tolist()

        self.original_code_list=get_original(original_code_path)


    def generate_example(self,batch_data,edge_batch,batch_index,batch_size):
        
        adv_source=[]
        adv_code=[]
        adv_nl=[]
       
       
        source_batch=batch_data[0]
        code_batch=batch_data[1]
        nl_batch=batch_data[2]
        edge_batch=edge_batch
        
        count=0
        SUBSTITUTION=3

        for source_list,n_list in zip(source_batch, nl_batch):
            
            source = source_list[0]
            for i in range(len(source_list) - 1):
                source= source + ' ' + source_list[i + 1]


            edge=edge_batch[count]
            code=self.original_code_list[(batch_index*batch_size)+count]

            variable_list = self.var_everyCode_list[(batch_index*batch_size)+count]
            sub_num=len(variable_list) if len(variable_list)<SUBSTITUTION else SUBSTITUTION
            masked_var_index=random.sample(range(0, len(variable_list)), sub_num)

            old_list=[]
            for item in masked_var_index:
                old_list.append(variable_list[item])
            

            new_source_list=[]

            for token in source_list:
                if token in old_list:
                    new_source_list.append('<UNK>')
                else:
                    new_source_list.append(token)

            adv_source.append(new_source_list)
            
            code_list=code.split(' ')
            new_code_list=[]
            for token in code_list:
                if token in old_list:
                    new_code_list.append('<UNK>')
                else:
                    new_code_list.append(token)


            sub_code_new_list=split_c_and_s2(new_code_list)
            adv_code.append(sub_code_new_list)

            adv_nl.append(n_list)

            count=count+1
       
        return [adv_source,adv_code,adv_nl]
        

