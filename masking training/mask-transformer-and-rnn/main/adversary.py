import pandas as pd
import torch
import numpy as np
import random
from c2nl.inputters.utils import process_examples
from c2nl.inputters import vector
from c2nl.inputters import dataset

from replace_and_camelSplit import split_c_and_s

def load_pkl(path):
    return pd.read_pickle(path)

class Adversary(object):
    def __init__(self,var_everyCode_path='../var_for_everyCode.pkl'):
        self.var_everyCode_list=load_pkl(var_everyCode_path)['variable'].tolist()
    



    def generate(self,exs,src_dict,tgt_dict):
        '''
            input: exs dict
            output: adv_exs  dict
        '''
        '''
            return {
            'ids': ids,  list
            'language': language,
            'batch_size': batch_size,
            'code_word_rep': code_word_rep,
            'code_char_rep': code_char_rep,
            'code_type_rep': code_type_rep,
            'code_mask_rep': code_mask_rep,
            'code_len': code_len_rep,
            'summ_word_rep': summ_word_rep,
            'summ_char_rep': summ_char_rep,
            'summ_len': summ_len_rep,
            'tgt_seq': tgt_tensor,
            'code_text': [ex['code'] for ex in batch],
            'code_tokens': [ex['code_tokens'] for ex in batch],
            'code_original':[ex['code_original'] for ex in batch],
            #################################################################
            'summ_text': [ex['summ'] for ex in batch],
            'summ_tokens': [ex['summ_tokens'] for ex in batch],
            'src_vocab': src_vocabs,
            'src_map': source_maps,
            'alignment': alignments,
            'stype': [ex['stype'] for ex in batch]
        }  
        '''


 
        LANG_ID=0  
        SOURCE_TAG=None
        MAX_SRC_LEN=150 
        MAX_TGT_LEN=50 
        CODE_TAG_TYPE='subtoken'
        uncase = True
        test_split = False
        batch_size=16
        data_workers=5
        cuda=True
        parallel=False


        SUBSTITUTION=3
        code_original_list=exs['code_original']
        ids=exs['ids']
        summ_list=exs['summ_text']

        unk_code_subtoken_list=[]
        examples = []
        for i in range(len(code_original_list)):
            code=code_original_list[i]
            code_id=ids[i]
            var_list=self.var_everyCode_list[code_id]

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

            sub_code_new=split_c_and_s(new_code_list)     #STR
            unk_code_subtoken_list.append(sub_code_new.strip())
        
        for count_id,src_sub,tgt,src_ori in zip(ids,unk_code_subtoken_list,summ_list,code_original_list):
            _ex = process_examples(LANG_ID,
                                   src_sub,
                                   SOURCE_TAG,
                                   tgt,
                                   src_ori,
                                   MAX_SRC_LEN,
                                   MAX_TGT_LEN,
                                   CODE_TAG_TYPE,
                                   count_id,
                                   uncase=uncase,
                                   test_split=False,
                                   )
                                
            if _ex is not None:
             
                examples.append(_ex)

        exs_dataset=dataset.CommentDataset_adv(examples,src_dict,tgt_dict)

        train_loader = torch.utils.data.DataLoader(
            exs_dataset,
            batch_size=batch_size,
            num_workers=data_workers,
            collate_fn=vector.batchify,
            pin_memory=cuda,
            drop_last=parallel,
            shuffle=False
        )

        for _,batch_data in enumerate(train_loader):
            adv_exs=batch_data
        return adv_exs


        

        
            



            


            



