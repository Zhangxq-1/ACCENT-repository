import pandas as pd
import time
SUBSTITUTION=2
def generate_adv():
    best_sub=pd.read_pickle('./var_name/java/test/test_java_IR_data.pkl')
    print(best_sub)
    best_sub_list=best_sub['var_sub'].tolist()
    
    

    f_code_origin=open('./var_name/java/code_ij.original','r')
    f_code_new=open('./var_name/java/test/test_java_code_IR_adv2.original','w')

    count=0
    for line in f_code_origin:
       
        best_sub_dict=best_sub_list[count]
        old_list=[]
        new_list=[]
        for k,v in best_sub_dict.items():
            old_list.append(k)
            new_list.append(v)
        if len(old_list)>SUBSTITUTION:
            old_list=old_list[0:SUBSTITUTION]
            new_list=new_list[0:SUBSTITUTION]
        code=line.split()
        new_code=[]
        for token in code:
            if token in old_list:
                index=old_list.index(token)
                new_code.append(new_list[index]) 
            else:
                new_code.append(token)

        line_new = new_code[0]
        for i in range(len(new_code) - 1):
            line_new = line_new + ' ' + new_code[i + 1]
        f_code_new.write(line_new+'\n')
        count=count+1

from replace_and_camelSplit import split_c_and_s
def generate_adv_subtoken():
    f=open('./var_name/java/test/test_java_code_IR_adv2.original','r')
    f_new=open('./rencos/var_name/java/test/test_java_code_IR_adv2.original_subtoken','w')

    for line in f:

        code=line.split(' ')
        line_new=split_c_and_s(code)
        f_new.write(line_new)
        print('ok')

generate_adv()
generate_adv_subtoken()


