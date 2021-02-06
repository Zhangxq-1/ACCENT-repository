import os
import torch
import pickle


class Dictionary(object):
    def __init__(self):
        self.word2idx = {'_UNK': 0}
        self.idx2word = ['_UNK']
        self.wordcnt = {'_UNK': 1}
        # self.word2idx = {}
        # self.idx2word = []
        # self.wordcnt = {}


    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
            self.wordcnt[word] = 1
        else:
            self.wordcnt[word] = self.wordcnt[word] + 1
        return self.word2idx[word]

    def getid(self, word, thresh=3):
        if (word not in self.word2idx) or (self.wordcnt[word] < thresh):
            return self.word2idx['_UNK']
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train_hebing.token.nl'))
        self.valid = self.tokenize(os.path.join(path, 'valid.token.nl'))
        self.test = self.tokenize(os.path.join(path, 'test.token.nl'))

        with open(os.path.join(path, 'dict_nl_hebing.pkl'), 'wb') as f:
            pickle.dump(self.dictionary, f)
        print (len(self.dictionary))
    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                if len(line.strip().split()) == 0:
                    continue
                words = ['<sos>'] + line.strip().split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                if len(line.strip().split()) == 0:
                    continue
                words = ['<sos>'] + line.strip().split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.getid(word)
                    token += 1

        return ids

