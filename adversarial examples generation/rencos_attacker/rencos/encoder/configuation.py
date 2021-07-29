class Configuration:
    def __init__(self):
        self.src_vocab_size = 24995 #31497  #24995-java
        self.target_vocab_size = 11649 #7682   #11649-java
        self.emb_dim = 64
        self.hid_dim = 512
        self.n_layers = 1
        self.dropout = 0.1
        self.learning_rate = 0.1
        self.momentum = 0.9
        self.max_length =300 #300  #150-java