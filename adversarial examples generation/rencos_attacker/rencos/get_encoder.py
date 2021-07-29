from encoder.rnnModel import Seq2Seq
import torch
import torch.nn as nn
import config
def pooling(x):
    x = torch.max(x, 1)[0]
    x=x.view(1,1,-1)

    out=nn.MaxPool1d(8,stride=8)
    x=out(x)
    x=x.squeeze()
    return x

def tensor_to_numpy(x):
    return x.detach().numpy()


def word2index(word,mode,vocab_src,vocab_trg,max_token_src,max_token_trg):
    word=word.split(' ')
    index_all_var=[]
    if mode == 'src':
        for item in word:
            if item in vocab_src:
                index_all_var.append(vocab_src[item].index)
            else:
                index_all_var.append(max_token_src)    #150 50
        if len(index_all_var)>=config.codeLen:
            index_all_var=index_all_var[0:config.codeLen]
        else:
            for i in range(config.codeLen-len(index_all_var)):
                index_all_var.append(max_token_src)
    else:
        for item in word:
            if item in vocab_trg:
                index_all_var.append(vocab_trg[item].index)
            else:
                index_all_var.append(max_token_trg)

        if len(index_all_var)>=config.summLen:
            index_all_var=index_all_var[0:config.summLen]
        else:
            for i in range(config.summLen-len(index_all_var)):
                index_all_var.append(max_token_trg)

    return index_all_var

def return_encoder(src,trg,model,vocab_src,vocab_trg,max_token_src,max_token_trg):

    src_index=[word2index(src,'src',vocab_src,vocab_trg,max_token_src,max_token_trg)]
    src_index = torch.LongTensor(src_index).view(config.codeLen, 1)   #150


    hidden, cell = model.encoder.forward(src_index)
    
    x=pooling(hidden)
    return tensor_to_numpy(x)




