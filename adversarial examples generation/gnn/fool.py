import pandas as pd

SUBSTITUTION=3
def generate_adv():
    best_sub=pd.read_pickle('/dataset/java/test/gnn_best_descend_data.pkl')
    best_sub_list=best_sub['var_sub'].tolist()

    f_code_origin=open('/dataset/java/test/code.token','r')
    f_code_new=open('/dataset/java/test/test_adv.token.code','w')

    count=0
    for line in f_code_origin:
        best_sub_dict=best_sub_list[count]
        print(best_sub_dict)
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
                new_code.append(new_list[index])  #xin de bian liang ming
            else:
                new_code.append(token)

        line_new = new_code[0]
        for i in range(len(new_code) - 1):
            line_new = line_new + ' ' + new_code[i + 1]
        f_code_new.write(line_new+'\n')
        count=count+1
        print('ok'+str(count))

from replace_and_camelSplit import split_c_and_s
def generate_adv_subtoken():
    f=open('/dataset/java/test/test_adv.token.code','r')
    f_new=open('/dataset/java/test/test_adv.token.code.subtoken','w')

    for line in f:

        code=line.split(' ')

        line_new=split_c_and_s(code)

        if '\n' in line_new:
            f_new.write(line_new)
        else:
            f_new.write(line_new+'\n')
       
    

generate_adv()
generate_adv_subtoken()

