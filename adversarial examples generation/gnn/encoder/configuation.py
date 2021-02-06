class Configuration:
    def __init__(self):
        self.src_vocab_size = 24995#15552  for python
        self.target_vocab_size = 11649#3442  for python
        self.emb_dim = 64
        self.hid_dim = 512
        self.n_layers = 1
        self.dropout = 0.1
        self.learning_rate = 0.1
        self.momentum = 0.9
        self.max_length = 150#400 for python