import javalang as javalang
import pandas as pd
import os
from utils import get_sequence as func

def trans_to_sequences(ast):
    sequence = []
    func(ast, sequence)
    return sequence

def extract_var_name(file,root,save_var_path,save_all_path):
    f_code = open(file, 'r')
    set_var = set()
    var_list = []
    index_list = []
    count = 0

    for line in f_code:

        code = line
        var = []
        #func name
        methodname_index = -1
        code_split = line.split(' ')
        for i in range(len(code_split)):
            if code_split[i] == '(':
                methodname_index = i - 1
                break
        if methodname_index == -1:
            print('can not find function name:' + str(count))
        else:
            var.append(code_split[methodname_index])
            set_var.add(code_split[methodname_index])

        tokens = javalang.tokenizer.tokenize(code)
        parser = javalang.parser.Parser(tokens)
        tree = parser.parse_member_declaration()
        ast_list = trans_to_sequences(tree)
        print(ast_list)

        for i in range(len(ast_list)):
            item = ast_list[i]
            if item == 'VariableDeclarator':
                var.append(ast_list[i + 1])
                set_var.add(ast_list[i + 1])

        # function name

        print(var)
        var_list.append(var)
        index_list.append(count)
        count = count + 1
        print('ok  ' + str(count))

    if not os.path.exists(root + '/var_name'):
        os.mkdir(root + '/var_name')

    data_var = pd.DataFrame({'id': index_list, 'variable': var_list})
    data_var.to_pickle(save_var_path)

    var_all_list = list(set_var)
    data_var_all = pd.DataFrame({'id': 0, 'all vars': [var_all_list]})
    data_var_all.to_pickle(save_all_path)

if __name__=='__main__':
    root=''
    file=root+'/data/python/test/code.original'
    save_all_path=root+'/var_for_allCode_test.pkl'
    save_var_path=root+'/var_for_everyCode_test.pkl'
    print('extract var name :')
    extract_var_name(file,root,save_var_path,save_all_path)
    print('extract end!')