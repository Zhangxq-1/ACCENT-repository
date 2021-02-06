import math
import torch
import pickle
import numpy as np
from torch.autograd import Variable




class LMProb():

    def __init__(self, model_path,dict_path):
        with open(model_path, 'rb') as f:
            self.model = torch.load(f)#, map_location={'cuda:0': 'cpu'})
            print(self.model)
            self.model.eval()
            self.model = self.model.cpu()
        with open(dict_path, 'rb') as f:
            self.dictionary = pickle.load(f)
        print (len(self.dictionary))
    def get_prob(self, words, verbose=False):
        pad_words = ['<sos>'] + words + ['<eos>']
        indxs = [self.dictionary.getid(w) for w in pad_words]
        # print (indxs, self.dictionary.getid('_UNK'))
        with torch.no_grad():
            #input = Variable(torch.LongTensor([int(indxs[0])]).unsqueeze(0), volatile=True)
            input = Variable(torch.LongTensor([int(indxs[0])]).unsqueeze(0))

        if verbose:
            print('words =', pad_words)
            print('indxs =', indxs)

        hidden = self.model.init_hidden(1)
        log_probs = []
        for i in range(1, len(pad_words)):
            output, hidden = self.model(input, hidden)
            # print (output.data.max(), output.data.exp())
            word_weights = output.squeeze().data.double().exp()
            # print (i, pad_words[i])
            prob = word_weights[indxs[i]] / word_weights.sum()
            log_probs.append(math.log(prob))
            # print('  {} => {:d},\tlogP(w|s)={:.4f}'.format(pad_words[i], indxs[i], log_probs[-1]))
            input.data.fill_(int(indxs[i]))

        if verbose:
            for i in range(len(log_probs)):
                print('  {} => {:d},\tlogP(w|s)={:.4f}'.format(pad_words[i+1], indxs[i+1], log_probs[i]))
            print('\n  => sum_prob = {:.4f}'.format(sum(log_probs)))

        # return sum(log_probs) / math.sqrt(len(log_probs))
        return sum(log_probs) / len(log_probs)







from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

lms = ['/data/java/code.pt','/data/java/nl.pt']
read_file_paths = ['/data/java/train.token.code', '/data/java/trai.token.nl']
dicts = ['/data/java/dict_code.pkl', '/data/java/dict_nl.pkl']
write_file_paths = ['/data/java/train.token.code.score', '/data/java/train.token.nl.score']
def get_score(line, num):
    sent = line.strip().split(' ')
    lm_score = lm_model.get_prob(sent)
    return (num, lm_score)

for i in range(2):
    lm_model = LMProb(lms[i], dicts[i])
    fw = open(write_file_paths[i], 'w')
    f = open(read_file_paths[i])
    lines = f.readlines()
    f.close()
    with ProcessPoolExecutor(max_workers=8) as executor:
        results = executor.map(get_score, lines, list(range(len(lines))))
    scores = {}
    for result in results:
        scores[result[0]] = result[1]
    for i in range(len(lines)):
        fw.write(str(scores[i]))
        fw.write('\n')
    fw.close()
