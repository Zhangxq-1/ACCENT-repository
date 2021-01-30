import javalang as javalang
import pandas as pd
import os
from utils import get_sequence as func

def trans_to_sequences(ast):
    sequence = []
    func(ast, sequence)
    return sequence

def extract_var_name(file,root,save_path):
    f_code = open(file, 'r')

    var_list = []
    index_list = []
    count = 0

    for line in f_code:

        code = line
        formalpara = []
        tokens = javalang.tokenizer.tokenize(code)
        parser = javalang.parser.Parser(tokens)
        tree = parser.parse_member_declaration()
        ast_list = trans_to_sequences(tree)

        for i in range(len(ast_list)):
            item = ast_list[i]
            if item == 'FormalParameter':
                formalpara.append(ast_list[i + 3])

        print(formalpara)
        var_list.append(formalpara)
        index_list.append(count)
        count = count + 1
        print('ok  ' + str(count))


    data_var = pd.DataFrame({'id': index_list, 'variable': var_list})
    data_var.to_pickle(save_path)


if __name__=='__main__':
    root=''
    file=root+'/data/python/test/code.original'
    save_path='/data/python/formalParameter_for_everyCode_test.pkl'
    print('extract var name :')
    extract_var_name(file,root,save_path)
    print('extract end!')