import pandas as pd
import torch
import numpy as np
import random


from onmt.replace_and_camelSplit import split_c_and_s
SUBSTITUTION=3
def load_pkl(path):
    return pd.read_pickle(path)
def load_file(code_path):
    code_list=[]
    f=open(code_path, 'r')
    for line in f:
        code_list.append(line)
    return code_list

class Adversary(object):
    def __init__(self,fields,max_sent_length,
    code_path='../samples/java/train/train.txt.src',
    var_everyCode_path='./java_var_for_everyCode.pkl'):
        self.var_everyCode_list=load_pkl(var_everyCode_path)['variable'].tolist()
        self.original_code_list=load_file(code_path)
        self.vocab=fields['src'].vocab.stoi
        self.max_len=max_sent_length


    def generate(self,i, batch_size):
     
        
        batch_code_list=self.original_code_list[i*batch_size:(i+1)*batch_size]
        tensor_list=[]

        batch_src_list=[]
        for idx in range(len(batch_code_list)):

            src_list=[]
            code=batch_code_list[idx]
            
            var_list=self.var_everyCode_list[idx+i*batch_size]

            sub_num=len(var_list) if len(var_list)<SUBSTITUTION else SUBSTITUTION
            masked_var_index=random.sample(range(0,len(var_list)),sub_num)
            old_list=[]
            for item in masked_var_index:
                old_list.append(var_list[item])


            code_list=code.split(' ')

            new_code_list=[]
            for token in code_list:
                if token in old_list:
                    new_code_list.append('<unk>')
                else:
                    new_code_list.append(token)

            sub_code_new=split_c_and_s(new_code_list) #str
#-----------------------------------------------------------------
            code_list=sub_code_new.strip().split(' ')
            src_list.append(self.vocab['<s>'])
            for item in code_list:
                if item in self.vocab.keys():

                    src_list.append(self.vocab[item])
                else:
                    src_list.append(self.vocab['<unk>'])
            if len(src_list) >self.max_len-1:
                src_list=src_list[0:self.max_len-1]
            else:
                dis=self.max_len-1-len(src_list)
                src_list=src_list+ [self.vocab['<blank>'] for i in range(0,dis)]

            src_list.append(self.vocab['</s>'])
            batch_src_list.append(src_list)

        src=torch.LongTensor(batch_src_list)
       
        src=src.reshape(self.max_len,batch_size,1).cuda() 
        return src


        
            



            


            



