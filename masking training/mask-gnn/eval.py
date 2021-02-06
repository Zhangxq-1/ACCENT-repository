import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import os
import time
import pickle
import models
import data
import utils
import config
import pandas as pd

class Eval(object):

    def __init__(self, model):

        # vocabulary
        self.source_vocab = utils.load_vocab_pk(config.source_vocab_path)
        self.source_vocab_size = len(self.source_vocab)
        self.code_vocab = utils.load_vocab_pk(config.code_vocab_path)
        self.code_vocab_size = len(self.code_vocab)
        self.nl_vocab = utils.load_vocab_pk(config.nl_vocab_path)
        self.nl_vocab_size = len(self.nl_vocab)
        #self.edages_data = pd.read_pickle(config.valid_edage_path)
        # dataset
        self.dataset = data.CodePtrDataset(source_path=config.valid_source_path,
                                           code_path=config.valid_code_path,
                                           nl_path=config.valid_nl_path)
        self.dataset_size = len(self.dataset)
        self.dataloader = DataLoader(dataset=self.dataset,
                                     batch_size=config.eval_batch_size,
                                     collate_fn=lambda *args: utils.collate_fn(args,
                                                                               source_vocab=self.source_vocab,
                                                                               code_vocab=self.code_vocab,
                                                                               nl_vocab=self.nl_vocab))

        # model
        if isinstance(model, str):
            self.model = models.Model(source_vocab_size=self.source_vocab_size,
                                      code_vocab_size=self.code_vocab_size,
                                      nl_vocab_size=self.nl_vocab_size,
                                      model_file_path=os.path.join(config.model_dir, model),
                                      is_eval=True)
        elif isinstance(model, dict):
            self.model = models.Model(source_vocab_size=self.source_vocab_size,
                                      code_vocab_size=self.code_vocab_size,
                                      nl_vocab_size=self.nl_vocab_size,
                                      model_state_dict=model,
                                      is_eval=True)
        else:
            raise Exception('Parameter \'model\' for class \'Eval\' must be file name or state_dict of the model.')

    def run_eval(self):
        loss = self.eval_iter()
        return loss
    
    def get_batch(self, edages_data, idx, bs):
        tmp = edages_data.iloc[idx*config.batch_size: idx*config.batch_size+bs]
        x1 = []
        for _, item in tmp.iterrows():
            x1.append(item['edge'])
        return x1

    def eval_one_batch(self, batch: utils.Batch, edge_batch, batch_size, criterion):
        """
        evaluate one batch
        :param batch:
        :param batch_size:
        :param criterion:
        :return:
        """
        with torch.no_grad():

            nl_batch = batch.nl_batch
            
            decoder_outputs = self.model(batch, batch_size, self.nl_vocab, edge_batch)  # [T, B, nl_vocab_size]

            batch_nl_vocab_size = decoder_outputs.size()[2]  # config.nl_vocab_size (+ max_oov_num)
            decoder_outputs = decoder_outputs.view(-1, batch_nl_vocab_size)
            nl_batch = nl_batch.view(-1)

            loss = criterion(decoder_outputs, nl_batch)
            
            return loss

    def eval_iter(self):
        """
        evaluate model on self.dataset
        :return: scores
        """
        epoch_loss = 0
        criterion = nn.NLLLoss(ignore_index=utils.get_pad_index(self.nl_vocab))

        for index_batch, batch in enumerate(self.dataloader):
            batch_size = batch.batch_size
            if index_batch == 0:
                f_edge = open(config.valid_edage_path,'rb')
                edages_data = pickle.load(f_edge)
                f_edge.close()

            edge_batch = self.get_batch(edages_data, index_batch, batch_size)

            loss = self.eval_one_batch(batch, edge_batch, batch_size, criterion=criterion)
            epoch_loss += loss.item()


        avg_loss = epoch_loss / len(self.dataloader)

        print('Validate completed, avg loss: {:.4f}.\n'.format(avg_loss))
        config.logger.info('Validate completed, avg loss: {:.4f}.'.format(avg_loss))

        return avg_loss

    def set_state_dict(self, state_dict):
        self.model.set_state_dict(state_dict)


class BeamNode(object):

    def __init__(self, sentence_indices, log_probs, hidden):
        """

        :param sentence_indices: indices of words of current sentence (from root to current node)
        :param log_probs: log prob of node of sentence
        :param hidden: [1, 1, H]
        """
        self.sentence_indices = sentence_indices
        self.log_probs = log_probs
        self.hidden = hidden

    def extend_node(self, word_index, log_prob, hidden):
        return BeamNode(sentence_indices=self.sentence_indices + [word_index],
                        log_probs=self.log_probs + [log_prob],
                        hidden=hidden)

    def avg_log_prob(self):
        return sum(self.log_probs) / len(self.sentence_indices)

    def word_index(self):
        return self.sentence_indices[-1]


class Test(object):

    def __init__(self, model):

        # vocabulary
        self.source_vocab = utils.load_vocab_pk(config.source_vocab_path)
        self.source_vocab_size = len(self.source_vocab)
        self.code_vocab = utils.load_vocab_pk(config.code_vocab_path)
        self.code_vocab_size = len(self.code_vocab)
        self.nl_vocab = utils.load_vocab_pk(config.nl_vocab_path)
        self.nl_vocab_size = len(self.nl_vocab)
        self.edages_data = pd.read_pickle(config.test_edage_path)

        # dataset
        self.dataset = data.CodePtrDataset(source_path=config.test_source_path,
                                           code_path=config.test_code_path,
                                           nl_path=config.test_nl_path)
        self.dataset_size = len(self.dataset)
        self.dataloader = DataLoader(dataset=self.dataset,
                                     batch_size=config.test_batch_size,
                                     collate_fn=lambda *args: utils.collate_fn(args,
                                                                               source_vocab=self.source_vocab,
                                                                               code_vocab=self.code_vocab,
                                                                               nl_vocab=self.nl_vocab,
                                                                               raw_nl=True))

        # model
        if isinstance(model, str):
            self.model = models.Model(source_vocab_size=self.source_vocab_size,
                                      code_vocab_size=self.code_vocab_size,
                                      nl_vocab_size=self.nl_vocab_size,
                                      model_file_path=os.path.join(config.model_dir, model),
                                      is_eval=True)
        elif isinstance(model, dict):
            self.model = models.Model(source_vocab_size=self.source_vocab_size,
                                      code_vocab_size=self.code_vocab_size,
                                      nl_vocab_size=self.nl_vocab_size,
                                      model_state_dict=model,
                                      is_eval=True)
        else:
            raise Exception('Parameter \'model\' for class \'Test\' must be file name or state_dict of the model.')

    def run_test(self) -> dict:
        """
        start test
        :return: scores dict, key is name and value is score
        """
        avg_s_bleu, avg_meteor,avg_rouge = self.test_iter()
        scores_dict = {
            's_bleu': avg_s_bleu,
            'meteor': avg_meteor,
            'rouge':avg_rouge
        }
        print('rouge:' + str(avg_rouge))
        utils.print_test_scores(scores_dict)
        return scores_dict
    
    def get_batch(self, idx, bs):
        tmp = self.edages_data.iloc[idx*config.batch_size: idx*config.batch_size+bs]
        x1 = []
        for _, item in tmp.iterrows():
            x1.append(item['edge'])
        return x1

    def test_one_batch(self, batch, edge_batch, batch_size):
        """

        :param batch:
        :param batch_size:
        :return:
        """
        with torch.no_grad():
            nl_batch = batch.nl_batch
            

            # outputs: [T, B, H]
            # hidden: [1, B, H]
            source_outputs, code_outputs, decoder_hidden = \
                self.model(batch, batch_size, self.nl_vocab, edge_batch, is_test=True)



            # decode
            batch_sentences = self.beam_decode(batch_size=batch_size,
                                               source_outputs=source_outputs,
                                               code_outputs=code_outputs,
                                               decoder_hidden=decoder_hidden)

            # translate indices into words both for candidates
            candidates = self.translate_indices(batch_sentences, batch.batch_oovs)

            # measure
            s_blue_score, meteor_score,rouge_score = utils.measure(batch_size, references=nl_batch, candidates=candidates)
            
            return nl_batch, candidates, s_blue_score, meteor_score,rouge_score

    def test_iter(self):
        """
        evaluate model on self.dataset
        :return: scores
        """
     
        start_time = time.time()
        total_references = []
        total_candidates = []
        total_s_bleu = 0
        total_meteor = 0
        total_rouge=0
        

        out_file = None
        if config.save_test_details:
            try:
                out_file = open(os.path.join(config.out_dir, 'test_details_{}.txt'.format(utils.get_timestamp())),
                                encoding='utf-8',
                                mode='w')
            except IOError:
                print('Test details file open failed.')

        sample_id = 0
        for index_batch, batch in enumerate(self.dataloader):
            batch_size = batch.batch_size
            edge_batch = self.get_batch(index_batch, batch_size)
   
            references, candidates, s_blue_score, meteor_score,rouge_score = self.test_one_batch(batch, edge_batch, batch_size)
          
            
            total_s_bleu += s_blue_score
            total_meteor += meteor_score
            total_rouge=total_rouge+rouge_score
            total_references += references
            total_candidates += candidates

            
            if index_batch % config.print_every == 0:
                cur_time = time.time()
                utils.print_test_progress(start_time=start_time, cur_time=cur_time, index_batch=index_batch,
                                          batch_size=batch_size, dataset_size=self.dataset_size,
                                          batch_s_bleu=s_blue_score, batch_meteor=meteor_score)


            if config.save_test_details:
                for index in range(len(references)):
                    '''
                    out_file.write('Sample {}:\n'.format(sample_id))
                    out_file.write(' '.join(['Reference:'] + references[index]) + '\n')
                    out_file.write(' '.join(['Candidate:'] + candidates[index]) + '\n')
                    out_file.write('\n')
                    '''
                    out_file.write(' '.join(references[index]) + '\n')
                    out_file.write(' '.join(candidates[index])+ '\n')
            
                    #sample_id += 1

        avg_s_bleu = total_s_bleu / self.dataset_size
        avg_meteor = total_meteor / self.dataset_size
        avg_rouge=total_rouge/ self.dataset_size

        return avg_s_bleu, avg_meteor,avg_rouge


    def beam_decode(self, batch_size, source_outputs: torch.Tensor, code_outputs: torch.Tensor,
                    decoder_hidden: torch.Tensor):
        """
        beam decode for one batch, feed one batch for decoder
        :param batch_size:
        :param source_outputs: [T, B, H]
        :param code_outputs: [T, B, H]
        :param ast_outputs: [T, B, H]
        :param decoder_hidden: [1, B, H]
        :param extend_source_batch: [B, T]
        :param extra_zeros: [B, max_oov_num]
        :return: batch_sentences, [B, config.beam_top_sentence]
        """
        batch_sentences = []

        for index_batch in range(batch_size):
            # for each input sentence
            single_decoder_hidden = decoder_hidden[:, index_batch, :].unsqueeze(1)  # [1, 1, H]
            single_source_output = source_outputs[:, index_batch, :].unsqueeze(1)   # [T, 1, H]
            single_code_output = code_outputs[:, index_batch, :].unsqueeze(1)  # [T, 1, H]



            root = BeamNode(sentence_indices=[utils.get_sos_index(self.nl_vocab)],
                            log_probs=[0.0],
                            hidden=single_decoder_hidden)

            current_nodes = [root]  # list of nodes to be further extended
            final_nodes = []  # list of end nodes

            for step in range(config.max_decode_steps):
                if len(current_nodes) == 0:
                    break

                candidate_nodes = []  # list of nodes to be extended next step

                feed_inputs = []
                feed_hidden = []
                feed_coverage = []

                # B = len(current_nodes) except eos
                extend_nodes = []
                for node in current_nodes:
                    # if current node is EOS
                    if node.word_index() == utils.get_eos_index(self.nl_vocab):
                        final_nodes.append(node)
                        # if number of final nodes reach the beam width
                        if len(final_nodes) >= config.beam_width:
                            break
                        continue

                    extend_nodes.append(node)

                    decoder_input = utils.tune_up_decoder_input(node.word_index(), self.nl_vocab)

                    single_decoder_hidden = node.hidden.clone().detach()     # [1, 1, H]

                    feed_inputs.append(decoder_input)  # [B]
                    feed_hidden.append(single_decoder_hidden)   # B x [1, 1, H]


                if len(extend_nodes) == 0:
                    break

                feed_batch_size = len(feed_inputs)
                feed_source_outputs = single_source_output.repeat(1, feed_batch_size, 1)
                feed_code_outputs = single_code_output.repeat(1, feed_batch_size, 1)

                

                feed_inputs = torch.tensor(feed_inputs, device=config.device)   # [B]
                feed_hidden = torch.stack(feed_hidden, dim=2).squeeze(0)    # [1, B, H]



                decoder_outputs, new_decoder_hidden, source_attn_weights, code_attn_weights = self.model.decoder(inputs=feed_inputs,
                                                       last_hidden=feed_hidden,
                                                       source_outputs=feed_source_outputs,
                                                       code_outputs=feed_code_outputs)

                # get top k words
                # log_probs: [B, beam_width]
                # word_indices: [B, beam_width]
                batch_log_probs, batch_word_indices = decoder_outputs.topk(config.beam_width)

                for index_node, node in enumerate(extend_nodes):
                    log_probs = batch_log_probs[index_node]
                    word_indices = batch_word_indices[index_node]
                    hidden = new_decoder_hidden[:, index_node, :].unsqueeze(1)


                    for i in range(config.beam_width):
                        log_prob = log_probs[i]
                        word_index = word_indices[i].item()

                        new_node = node.extend_node(word_index=word_index,
                                                    log_prob=log_prob,
                                                    hidden=hidden)
                        candidate_nodes.append(new_node)

                # sort candidate nodes by log_prb and select beam_width nodes
                candidate_nodes = sorted(candidate_nodes, key=lambda item: item.avg_log_prob(), reverse=True)
                current_nodes = candidate_nodes[: config.beam_width]

            final_nodes += current_nodes
            final_nodes = sorted(final_nodes, key=lambda item: item.avg_log_prob(), reverse=True)
            final_nodes = final_nodes[: config.beam_top_sentences]

            sentences = []
            for final_node in final_nodes:
                sentences.append(final_node.sentence_indices)

            batch_sentences.append(sentences)

        return batch_sentences

    def translate_indices(self, batch_sentences, batch_oovs: list):
        """
        translate indices to words for one batch
        :param batch_sentences: [B, config.beam_top_sentences, sentence_length]
        :param batch_oovs: list of oov words list for one batch, None if not use pointer gen, [B, oov_num(variable)]
        :return:
        """
        batch_words = []
        for index_batch, sentences in enumerate(batch_sentences):
            words = []
            for indices in sentences:
                for index in indices:   # indices is a list of length 1, only loops once
                    if index not in self.nl_vocab.index2word:   # current index is out of vocabulary
                        assert batch_oovs is not None       # should not happen when not use pointer gen
                        oovs = batch_oovs[index_batch]      # oov list for current sample
                        oov_index = index - self.nl_vocab_size  # oov temp index
                        try:
                            word = oovs[oov_index]
                            config.logger.info('Pointed OOV word: {}'.format(word))
                        except IndexError:
                            # raise IndexError('Error: model produced word id', index,
                            #                  'which is corresponding to an OOV word index', oov_index,
                            #                  'but this sample only has {} OOV words.'.format(len(oovs)))
                            word = '<UNK>'
                    else:
                        word = self.nl_vocab.index2word[index]
                    if utils.is_unk(word) or not utils.is_special_symbol(word):
                        words.append(word)
            batch_words.append(words)
        return batch_words


#
#
#
#
#

class Test_adv(object):

    def __init__(self,source_vocab,code_vocab,nl_vocab):

        # vocabulary
        self.source_vocab = source_vocab
        self.source_vocab_size = len(self.source_vocab)
        self.code_vocab =code_vocab
        self.code_vocab_size = len(self.code_vocab)
        self.nl_vocab = nl_vocab
        self.nl_vocab_size = len(self.nl_vocab)
        self.model=None


    def init_model(self,model):
        self.model=model
        

    #----------------- init dataloader for adversarial ----------------------------------------
    def init_dataloader(self,source, code, nl):
        self.dataset=data.CodePtrDataset_1(source,code,nl)
        self.dataloader = DataLoader(dataset=self.dataset,
                                     batch_size=1,
                                     collate_fn=lambda *args: utils.collate_fn(args,
                                                                               source_vocab=self.source_vocab,
                                                                               code_vocab=self.code_vocab,
                                                                               nl_vocab=self.nl_vocab,
                                                                               raw_nl=True))

    #-------------------------------test for adversarial ------------------------------
    def test_for_adv(self,edge_batch):
        for index_batch, batch in enumerate(self.dataloader):
            batch_size = batch.batch_size
            #edge_batch = self.get_batch(index_batch, batch_size)   #  direct pass to edge_batch
            source_batch, source_seq_lens, code_batch, code_seq_lens, \
            nl_batch, nl_seq_lens = batch.get_regular_input()
            '''
            print(source_batch.shape)
            print(code_batch.shape)
            print(len(edge_batch))
            print(batch_size)
            '''
            candidates, encoder= self.test_one_batch(batch, edge_batch, batch_size)
            return candidates, encoder
          
    #----------------------------- not used -----------------------------------------------------

   

    def test_one_batch(self, batch, edge_batch, batch_size):
        """

        :param batch:
        :param batch_size:
        :return:
        """
        with torch.no_grad():

            # outputs: [T, B, H]
            # hidden: [1, B, H]
            source_outputs, code_outputs, decoder_hidden, = \
                self.model(batch, batch_size, self.nl_vocab, edge_batch, is_test=True)
            '''  
            print('----------------')
            print(source_outputs.shape)
            print(code_outputs.shape)
            print('--------------')
            '''
            # decode
            batch_sentences = self.beam_decode(batch_size=batch_size,
                                               source_outputs=source_outputs,
                                               code_outputs=code_outputs,
                                               decoder_hidden=decoder_hidden)

            # translate indices into words both for candidates
            candidates = self.translate_indices(batch_sentences, batch.batch_oovs)

            # measure
            #s_blue_score, _ = utils.measure(batch_size, references=nl_batch, candidates=candidates)
            #return nl_batch, candidates, s_blue_score, meteor_score
            return candidates,[source_outputs,code_outputs]   #encoder
            #return s_blue_score,[source_outputs,code_outputs]


    
    def beam_decode(self, batch_size, source_outputs: torch.Tensor, code_outputs: torch.Tensor,
                    decoder_hidden: torch.Tensor):
        """
        beam decode for one batch, feed one batch for decoder
        :param batch_size:
        :param source_outputs: [T, B, H]
        :param code_outputs: [T, B, H]
        :param ast_outputs: [T, B, H]
        :param decoder_hidden: [1, B, H]
        :param extend_source_batch: [B, T]
        :param extra_zeros: [B, max_oov_num]
        :return: batch_sentences, [B, config.beam_top_sentence]
        """
        batch_sentences = []

        for index_batch in range(batch_size):
            # for each input sentence
            single_decoder_hidden = decoder_hidden[:, index_batch, :].unsqueeze(1)  # [1, 1, H]
            single_source_output = source_outputs[:, index_batch, :].unsqueeze(1)   # [T, 1, H]
            single_code_output = code_outputs[:, index_batch, :].unsqueeze(1)  # [T, 1, H]



            root = BeamNode(sentence_indices=[utils.get_sos_index(self.nl_vocab)],
                            log_probs=[0.0],
                            hidden=single_decoder_hidden)

            current_nodes = [root]  # list of nodes to be further extended
            final_nodes = []  # list of end nodes

            for step in range(config.max_decode_steps):
                if len(current_nodes) == 0:
                    break

                candidate_nodes = []  # list of nodes to be extended next step

                feed_inputs = []
                feed_hidden = []
                feed_coverage = []

                # B = len(current_nodes) except eos
                extend_nodes = []
                for node in current_nodes:
                    # if current node is EOS
                    if node.word_index() == utils.get_eos_index(self.nl_vocab):
                        final_nodes.append(node)
                        # if number of final nodes reach the beam width
                        if len(final_nodes) >= config.beam_width:
                            break
                        continue

                    extend_nodes.append(node)

                    decoder_input = utils.tune_up_decoder_input(node.word_index(), self.nl_vocab)

                    single_decoder_hidden = node.hidden.clone().detach()     # [1, 1, H]

                    feed_inputs.append(decoder_input)  # [B]
                    feed_hidden.append(single_decoder_hidden)   # B x [1, 1, H]


                if len(extend_nodes) == 0:
                    break

                feed_batch_size = len(feed_inputs)
                feed_source_outputs = single_source_output.repeat(1, feed_batch_size, 1)
                feed_code_outputs = single_code_output.repeat(1, feed_batch_size, 1)

                

                feed_inputs = torch.tensor(feed_inputs, device=config.device)   # [B]
                feed_hidden = torch.stack(feed_hidden, dim=2).squeeze(0)    # [1, B, H]



                decoder_outputs, new_decoder_hidden, source_attn_weights, code_attn_weights = self.model.decoder(inputs=feed_inputs,
                                                       last_hidden=feed_hidden,
                                                       source_outputs=feed_source_outputs,
                                                       code_outputs=feed_code_outputs)

                # get top k words
                # log_probs: [B, beam_width]
                # word_indices: [B, beam_width]
                batch_log_probs, batch_word_indices = decoder_outputs.topk(config.beam_width)

                for index_node, node in enumerate(extend_nodes):
                    log_probs = batch_log_probs[index_node]
                    word_indices = batch_word_indices[index_node]
                    hidden = new_decoder_hidden[:, index_node, :].unsqueeze(1)


                    for i in range(config.beam_width):
                        log_prob = log_probs[i]
                        word_index = word_indices[i].item()

                        new_node = node.extend_node(word_index=word_index,
                                                    log_prob=log_prob,
                                                    hidden=hidden)
                        candidate_nodes.append(new_node)

                # sort candidate nodes by log_prb and select beam_width nodes
                candidate_nodes = sorted(candidate_nodes, key=lambda item: item.avg_log_prob(), reverse=True)
                current_nodes = candidate_nodes[: config.beam_width]

            final_nodes += current_nodes
            final_nodes = sorted(final_nodes, key=lambda item: item.avg_log_prob(), reverse=True)
            final_nodes = final_nodes[: config.beam_top_sentences]

            sentences = []
            for final_node in final_nodes:
                sentences.append(final_node.sentence_indices)

            batch_sentences.append(sentences)

        return batch_sentences

    def translate_indices(self, batch_sentences, batch_oovs: list):
        """
        translate indices to words for one batch
        :param batch_sentences: [B, config.beam_top_sentences, sentence_length]
        :param batch_oovs: list of oov words list for one batch, None if not use pointer gen, [B, oov_num(variable)]
        :return:
        """
        batch_words = []
        for index_batch, sentences in enumerate(batch_sentences):
            words = []
            for indices in sentences:
                for index in indices:   # indices is a list of length 1, only loops once
                    if index not in self.nl_vocab.index2word:   # current index is out of vocabulary
                        assert batch_oovs is not None       # should not happen when not use pointer gen
                        oovs = batch_oovs[index_batch]      # oov list for current sample
                        oov_index = index - self.nl_vocab_size  # oov temp index
                        try:
                            word = oovs[oov_index]
                            config.logger.info('Pointed OOV word: {}'.format(word))
                        except IndexError:
                            # raise IndexError('Error: model produced word id', index,
                            #                  'which is corresponding to an OOV word index', oov_index,
                            #                  'but this sample only has {} OOV words.'.format(len(oovs)))
                            word = '<UNK>'
                    else:
                        word = self.nl_vocab.index2word[index]
                    if utils.is_unk(word) or not utils.is_special_symbol(word):
                        words.append(word)
            batch_words.append(words)
        return batch_words
