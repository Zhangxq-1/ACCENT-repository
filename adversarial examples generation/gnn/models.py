import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import math
import random

import config
import utils



class Encoder(nn.Module):
    """
    Encoder for both code and ast
    """

    def __init__(self, vocab_size):
        super(Encoder, self).__init__()
        self.hidden_size = config.hidden_size
        self.num_directions = 2

        # vocab_size: config.code_vocab_size for code encoder, size of sbt vocabulary for ast encoder
        self.embedding = nn.Embedding(vocab_size, config.embedding_dim)
        self.gru = nn.GRU(config.embedding_dim, self.hidden_size, bidirectional=True)


    def forward(self, inputs: torch.Tensor, seq_lens: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """

        :param inputs: sorted by length in descending order, [T, B]
        :param seq_lens: should be in descending order
        :return: outputs: [T, B, H]
                hidden: [2, B, H]
        """
        embedded = self.embedding(inputs)   # [T, B, embedding_dim]
        packed = pack_padded_sequence(embedded, seq_lens, enforce_sorted=False)
        outputs, hidden = self.gru(packed)
        outputs, _ = pad_packed_sequence(outputs)  # [T, B, 2*H]
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]

        # outputs: [T, B, H]
        # hidden: [2, B, H]
        return outputs, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_directions, batch_size, self.hidden_size, device=config.device)


class GCNEncoder(nn.Module):
    """
    Encoder for both code and ast
    """

    def __init__(self, vocab_size):
        super(GCNEncoder, self).__init__()
        self.hidden_size = config.hidden_size
        self.num_directions = 2

        # vocab_size: config.code_vocab_size for code encoder, size of sbt vocabulary for ast encoder
        self.embedding = nn.Embedding(vocab_size, config.embedding_dim)
        self.fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.gru = nn.GRU(config.embedding_dim, self.hidden_size, bidirectional=True)


    def forward(self, inputs: torch.Tensor, seq_lens: torch.Tensor, adjacent: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """

        :param inputs: sorted by length in descending order, [T, B]
        :param seq_lens: should be in descending order
        :return: outputs: [T, B, H]
                hidden: [2, B, H]
        """
        embedded0 = self.embedding(inputs)   # [T, B, embedding_dim]
        embedded0 = embedded0.transpose(0,1)  #[B,T,H]
        _, size_em, _ = embedded0.size()
        embedded = torch.zeros(1,201,config.embedding_dim).cuda()
        embedded[:,:size_em,:] = embedded0
        adjacents = torch.Tensor(adjacent).cuda()
        adj = torch.zeros(1,201,201).cuda()
        adj = adjacents[:,:201,:201]
        adj[:,200,200] = 0
        #adj = torch.Tensor(adjacent).cuda()
        
        edges1 = torch.bmm(adj, embedded) #[B,T,H]
        outs1 = self.fc1(edges1) 
        out1 = F.relu(outs1)
        edges2 = torch.bmm(adj, out1) #[B,T,H]
        outs2 = self.fc2(edges2) 
        out2 = F.relu(outs2)

        embedded = out2.transpose(0,1)
        packed = pack_padded_sequence(embedded, seq_lens, enforce_sorted=False)
        outputs, hidden = self.gru(packed)
        outputs, _ = pad_packed_sequence(outputs)  # [T, B, 2*H]
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]

        # outputs: [T, B, H]
        # hidden: [2, B, H]
        return outputs, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_directions, batch_size, self.hidden_size, device=config.device)

class Attention(nn.Module):

    def __init__(self, hidden_size=config.hidden_size):
        super(Attention, self).__init__()
        
        self.hidden_size = hidden_size

        self.attn = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.v = nn.Parameter(torch.rand(self.hidden_size), requires_grad=True)   # [H]
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)

    def forward(self, hidden, encoder_outputs):
        """
        forward the net
        :param hidden: the last hidden state of encoder, [1, B, H]
        :param encoder_outputs: [T, B, H]
        :param coverage: last coverage vector, [B, T]
        :return: softmax scores, [B, 1, T]
        """
        time_step, batch_size, _ = encoder_outputs.size()
        
        h = hidden.repeat(time_step, 1, 1).transpose(0, 1)  # [B, T, H]
        encoder_outputs = encoder_outputs.transpose(0, 1)   # [B, T, H]

        attn_energies = self.score(h, encoder_outputs)      # [B, T]
        attn_weights = F.softmax(attn_energies, dim=1).unsqueeze(1)     # [B, 1, T]

        return attn_weights

    def score(self, hidden, encoder_outputs):
        """
        calculate the attention scores of each word
        :param hidden: [B, T, H]
        :param encoder_outputs: [B, T, H]
        :param coverage: [B, T]
        :return: energy: scores of each word in a batch, [B, T]
        """
        # after cat: [B, T, 2/3*H]
        # after attn: [B, T, H]
        # energy: [B, T, H]
        energy = F.relu(self.attn(torch.cat([hidden, encoder_outputs], dim=2)))     # [B, T, H]
        energy = energy.transpose(1, 2)     # [B, H, T]
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)      # [B, 1, H]
        energy = torch.bmm(v, energy)   # [B, 1, T]
        return energy.squeeze(1)


class Decoder(nn.Module):

    def __init__(self, vocab_size, hidden_size=config.hidden_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, config.embedding_dim)
        self.dropout = nn.Dropout(config.decoder_dropout_rate)
        self.source_attention = Attention()
        self.code_attention = Attention()
        self.gru = nn.GRU(config.embedding_dim + self.hidden_size, self.hidden_size)
        self.out = nn.Linear(2 * self.hidden_size, config.nl_vocab_size)



    def forward(self, inputs, last_hidden, source_outputs, code_outputs):
        
        embedded = self.embedding(inputs).unsqueeze(0)      # [1, B, embedding_dim]
        # embedded = self.dropout(embedded)

        # get attn weights of source
        # calculate and add source context in order to update attn weights during training
        source_attn_weights = self.source_attention(last_hidden, source_outputs)  # [B, 1, T]
        source_context = source_attn_weights.bmm(source_outputs.transpose(0, 1))  # [B, 1, H]
        source_context = source_context.transpose(0, 1)  # [1, B, H]

        code_attn_weights = self.code_attention(last_hidden, code_outputs)  # [B, 1, T]
        code_context = code_attn_weights.bmm(code_outputs.transpose(0, 1))  # [B, 1, H]
        code_context = code_context.transpose(0, 1)     # [1, B, H]

       # make ratio between source code and construct is 1: 1
        context = 0.5*source_context + code_context   # [1, B, H]



        rnn_input = torch.cat([embedded, context], dim=2)   # [1, B, embedding_dim + H]
        outputs, hidden = self.gru(rnn_input, last_hidden)  # [1, B, H] for both

        outputs = outputs.squeeze(0)    # [B, H]
        context = context.squeeze(0)    # [B, H]

        vocab_dist = self.out(torch.cat([outputs, context], 1))    # [B, nl_vocab_size]
        vocab_dist = F.softmax(vocab_dist, dim=1)     # P_vocab, [B, nl_vocab_size]

        final_dist = vocab_dist

        final_dist = torch.log(final_dist + config.eps)

        return final_dist, hidden, source_attn_weights, code_attn_weights


class Model(nn.Module):

    def __init__(self, source_vocab_size, code_vocab_size, nl_vocab_size,
                 model_file_path=None, model_state_dict=None, is_eval=False):
        super(Model, self).__init__()

        # vocabulary size for encoders
        self.source_vocab_size = source_vocab_size
        self.code_vocab_size = code_vocab_size
        self.is_eval = is_eval

        # init models
        self.source_encoder = GCNEncoder(self.source_vocab_size)
        self.code_encoder = Encoder(self.code_vocab_size)
        self.decoder = Decoder(nl_vocab_size)

        if config.use_cuda:
            self.source_encoder = self.source_encoder.cuda()
            self.code_encoder = self.code_encoder.cuda()
            self.decoder = self.decoder.cuda()

        if model_file_path:
            state = torch.load(model_file_path)
            self.set_state_dict(state)

        if model_state_dict:
            self.set_state_dict(model_state_dict)

        if is_eval:
            self.source_encoder.eval()
            self.code_encoder.eval()
            self.decoder.eval()

    def forward(self, batch, batch_size, nl_vocab, adjacent, is_test=False):
        """

        :param batch:
        :param batch_size:
        :param nl_vocab:
        :param is_test: if True, function will return before decoding
        :return: decoder_outputs: [T, B, nl_vocab_size]
        """
        # batch: [T, B]
        source_batch, source_seq_lens, code_batch, code_seq_lens, \
            nl_batch, nl_seq_lens = batch.get_regular_input()

        # encode
        # outputs: [T, B, H]
        # hidden: [2, B, H]
        source_outputs, source_hidden = self.source_encoder(source_batch, source_seq_lens, adjacent)
        code_outputs, code_hidden = self.code_encoder(code_batch, code_seq_lens)
        
        
        code_hidden = code_hidden[:1]  # [1, B, H]
        decoder_hidden = code_hidden
        '''
        source_hidden = source_hidden[:1]
        decoder_hidden = source_hidden
        '''


        if is_test:
            return source_outputs, code_outputs, code_hidden

        if nl_seq_lens is None:
            max_decode_step = config.max_decode_steps
        else:
            max_decode_step = max(nl_seq_lens)

        decoder_inputs = utils.init_decoder_inputs(batch_size=batch_size, vocab=nl_vocab)  # [B]

        decoder_outputs = torch.zeros((max_decode_step, batch_size, config.nl_vocab_size), device=config.device)

        for step in range(max_decode_step):
            # decoder_outputs: [B, nl_vocab_size]
            # decoder_hidden: [1, B, H]
            # attn_weights: [B, 1, T]
            decoder_output, decoder_hidden, source_attn_weights, code_attn_weights = self.decoder(inputs=decoder_inputs,
                                             last_hidden=decoder_hidden,
                                             source_outputs=source_outputs,
                                             code_outputs=code_outputs)
            decoder_outputs[step] = decoder_output

            if config.use_teacher_forcing and random.random() < config.teacher_forcing_ratio and not self.is_eval:
                # use teacher forcing, ground truth to be the next input
                decoder_inputs = nl_batch[step]
            else:
                # output of last step to be the next input
                _, indices = decoder_output.topk(1)  # [B, 1]
                decoder_inputs = indices.squeeze(1).detach()  # [B]
                decoder_inputs = decoder_inputs.to(config.device)
        return decoder_outputs

    def set_state_dict(self, state_dict):
        self.source_encoder.load_state_dict(state_dict["source_encoder"])
        self.code_encoder.load_state_dict(state_dict["code_encoder"])
        self.decoder.load_state_dict(state_dict["decoder"])
