from attacker.encoder.rnnModel import *
from attacker.encoder.configuation import *
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from gensim.models.word2vec import Word2Vec
import pandas as pd

# init weights
def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)


def get_batch(dataset,idx,bs):
    tmp=dataset.iloc[idx:idx+bs]
    src=[]
    trg=[]
    for _,item in tmp.iterrows():
        code=item['code']
        code_index=word2index(code,'src')
        src.append(code_index)

        summ=item['summ']
        summ_index=word2index(summ,'trg')
        trg.append(summ_index)
    src = torch.LongTensor(src).view(150,32)

    trg=torch.LongTensor(trg).view(50,32)
    return src,trg

def word2index(word,mode):
    word=word.split(' ')
    index_all_var=[]
    if mode == 'src':
        for item in word:
            if item in vocab_src:
                index_all_var.append(vocab_src[item].index)
            else:
                index_all_var.append(max_token_src)
        if len(index_all_var)>=150:
            index_all_var=index_all_var[0:150]
        else:
            for i in range(150-len(index_all_var)):
                index_all_var.append(max_token_src)
    else:
        for item in word:
            if item in vocab_trg:
                index_all_var.append(vocab_trg[item].index)
            else:
                index_all_var.append(max_token_trg)

        if len(index_all_var)>=50:
            index_all_var=index_all_var[0:50]
        else:
            for i in range(50-len(index_all_var)):
                index_all_var.append(max_token_trg)

    return index_all_var



# optimizer


def train(model, train_data, optimizer, criterion):
    model.train()

    for epoch in range(EPOCHES):
        epoch_loss = 0
        old_loss=10000.0
        i=0
        while i <len(train_data):
            batch = get_batch(train_data,i,batch_size)
            i=i+batch_size
            src,trg=batch

            if USE_GPU:
                src,trg=src.cuda(),trg.cuda()
            model.zero_grad()
            output,_ = model.forward(src, trg)
            output = output.view(-1, output.shape[-1])
            trg = trg.view(-1)

            # output = [(trg sent len - 1) * batch size, output dim]
            # trg = [(trg sent len - 1) * batch size]

            loss = criterion(output, trg)

            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()
        avg_loss=epoch_loss/len(train_data)
        print('-------------------training loss of epoch ' +str(epoch) +':------------------'+str(avg_loss))
        if avg_loss < old_loss:
            torch.save(model.state_dict(),'/encoder/model.pkl')
            old_loss=avg_loss





if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data=pd.read_pickle('/encoder/train_data.pkl')
    print('read embedding....')
    root = ' '
    word2vec_src = Word2Vec.load(root + '/vocab/train/node_w2v_code_64').wv
    vocab_src = word2vec_src.vocab
    max_token_src = word2vec_src.syn0.shape[0]
    embedding_dim = word2vec_src.syn0.shape[1]
    embeddings_src = np.zeros((max_token_src + 1, embedding_dim))
    embeddings_src[:max_token_src] = word2vec_src.syn0

    word2vec_trg = Word2Vec.load(root + '/vocab/train/node_w2v_summ_64').wv
    vocab_trg = word2vec_trg.vocab
    max_token_trg = word2vec_trg.syn0.shape[0]
    embedding_dim = word2vec_trg.syn0.shape[1]
    embeddings_trg = np.zeros((max_token_trg + 1, embedding_dim))
    embeddings_trg[:max_token_trg] = word2vec_trg.syn0

    src_vocab=embeddings_src
    target_vocab=embeddings_trg
    EPOCHES=100
    batch_size=32
    USE_GPU=True

    model = Seq2Seq(src_vocab, target_vocab).to(device)
    model.apply(init_weights)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    train(model,data,optimizer,criterion)