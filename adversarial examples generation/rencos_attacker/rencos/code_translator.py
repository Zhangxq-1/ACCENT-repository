
""" Translator Class and builder """
import argparse
import codecs
import os
import torch

from itertools import count
from onmt.utils.misc import tile
import numpy as np

import onmt.model_builder
import onmt.translate.beam
import onmt.inputters as inputters
import onmt.opts as opts
import onmt.decoders.ensemble
from onmt.translate.translator import Translator
import config

def build_translator(opt, report_score=True, logger=None, out_file=None):
    if out_file is None:
        out_file = codecs.open(opt.output, 'w+', 'utf-8') 

    if opt.gpu > -1:
        torch.cuda.set_device(opt.gpu)

    dummy_parser = argparse.ArgumentParser(description='train.py')
    opts.model_opts(dummy_parser)
    dummy_opt = dummy_parser.parse_known_args([])[0]

    if len(opt.models) > 1:
        # use ensemble decoding if more than one model is specified
        fields, model, model_opt = \
            onmt.decoders.ensemble.load_test_model(opt, dummy_opt.__dict__)
    else:
        fields, model, model_opt = \
            onmt.model_builder.load_test_model(opt, dummy_opt.__dict__)
     
   
   
    scorer = onmt.translate.GNMTGlobalScorer(opt)


    translator = CodeTranslator(model, fields, opt, model_opt,
                            global_scorer=scorer, out_file=out_file,
                            report_score=report_score, logger=logger)
    translator.src_path = opt.src  
    return translator


class CodeTranslator(Translator):
    def data_data_iter_builder(self,
                  src_path=None,
                  src_data_iter=None,
                  src_length=None,
                  tgt_path=None,
                  tgt_data_iter=None,
                  src_dir=None,
                  batch_size=None,
                  attn_debug=False, search_mode=0, threshold=0,
                  ref_path=None):
        
        assert src_data_iter is not None or src_path is not None
        if batch_size is None:
            raise ValueError("batch_size must be set")
        data = inputters.build_dataset(self.fields,
                                       self.data_type,
                                       src_path=src_path,
                                       src_data_iter=src_data_iter,
                                       src_seq_length_trunc=src_length,
                                       tgt_path=tgt_path,
                                       tgt_data_iter=tgt_data_iter,
                                       src_dir=src_dir,
                                       sample_rate=self.sample_rate,
                                       window_size=self.window_size,
                                       window_stride=self.window_stride,
                                       window=self.window,
                                       use_filter_pred=self.use_filter_pred,
                                       ref_path=['%s.%d'%(ref_path, r) for r in range(self.refer)] if self.refer else None,
                                       ref_seq_length_trunc=self.max_sent_length,
									   ignore_unk=False)

        if self.cuda:
            cur_device = "cuda"
        else:
            cur_device = "cpu"
        if self.refer:
            for i in range(self.refer):
                data.fields['ref%d'%i].vocab = data.fields['src'].vocab

        data_iter = inputters.OrderedIterator(
            dataset=data, device=cur_device,
            batch_size=batch_size, train=False, sort=False,
            sort_within_batch=True, shuffle=False)
       
        builder = onmt.translate.TranslationBuilder(
            data, self.fields,
            self.n_best, self.replace_unk, tgt_path)
       
        return data, data_iter, builder

    def transToSent(self,builder, batch_data, tgt_path=None):   
        translations = builder.from_batch(batch_data)
        all_scores = []
        all_predictions = []
        pred_score_total, pred_words_total = 0, 0
        gold_score_total, gold_words_total = 0, 0
        for trans in translations:
            all_scores += [trans.pred_scores[:self.n_best]]
            pred_score_total += trans.pred_scores[0]
            pred_words_total += len(trans.pred_sents[0])

            if tgt_path is not None:
                gold_score_total += trans.gold_score
                gold_words_total += len(trans.gold_sent) + 1

            n_best_preds = [" ".join(pred)
                                for pred in trans.pred_sents[:self.n_best]]
            all_predictions += [n_best_preds]
              
 
      
        return all_scores, all_predictions
    

    def translate_batch_adv(self, batch, data, first,attn_debug=False,ori_data_token='', fast=True):
       
        with torch.no_grad():
           
            return self._fast_translate_batch_adv(
                    batch,
                    data,
                    self.max_length,first,
                    min_length=self.min_length,ori_data_token='',
                    n_best=self.n_best,
                    return_attention=attn_debug or self.replace_unk)
    
    def _fast_translate_batch_adv(self,
                              batch,
                              data,
                              max_length,first,
                              min_length=0,ori_data_token='',
                              n_best=1,  #!!!!!!!!!!!!!记得其他时候 first变成false
                              return_attention=False):
        # TODO: faster code path for beam_size == 1.

        # TODO: support these blacklisted features.
        assert not self.dump_beam
        assert not self.use_filter_pred
        assert self.block_ngram_repeat == 0
        assert self.global_scorer.beta == 0

        
        beam_size=1
        batch_size = batch.batch_size
        vocab = self.fields["tgt"].vocab
        start_token = vocab.stoi[inputters.BOS_WORD]
        end_token = vocab.stoi[inputters.EOS_WORD]
        print(first)
        # Encoder forward.
        if first:
            src, enc_states, memory_bank, src_lengths = self._run_encoder(
                batch, data.data_type)
            self.model.decoder.init_state(src, memory_bank, enc_states, with_cache=True)
        else:
          
            
            vocab_src_adv=self.fields['src'].vocab.stoi
           
            src_list=[]
            code_list=ori_data_token.strip().split(' ')
            src_list.append(vocab_src_adv['<s>'])
            for item in code_list:
                if item in vocab_src_adv.keys():

                    src_list.append(vocab_src_adv[item])
                else:
                    src_list.append(vocab_src_adv['<unk>'])
            if len(src_list) >config.len_sent-1:
                src_list=src_list[0:config.len_sent-1]
            else:
                dis=config.len_sent-1-len(src_list)
                src_list=src_list+ [vocab_src_adv['<blank>'] for i in range(0,dis)]

            src_list.append(vocab_src_adv['</s>'])
            src=torch.LongTensor([src_list])
            src=src.reshape(config.len_sent,1,1).cuda() 
            src_lengths=torch.LongTensor([[len(code_list)]]).reshape(1).cuda()
            
            enc_states, memory_bank, src_lengths = self.model.encoder(
            src, src_lengths)
            self.model.decoder.init_state(src, memory_bank, enc_states, with_cache=True)

        if first:
            ref_list, ref_states_list, ref_bank_list, ref_lengths_list, ref_prs_list = [], [], [], [], []
            for i in range(self.refer):
                ref, ref_states, ref_bank, ref_lengths, ref_prs = self._run_refer(batch, data.data_type, k=i)
                ref_list.append(ref)
                ref_states_list.append(ref_states)
                ref_bank_list.append(ref_bank)
                ref_lengths_list.append(ref_lengths)
                ref_prs_list.append(ref_prs)
                self.extra_decoders[i].init_state(ref, ref_bank, ref_states, with_cache=True)

            self.ref_list=ref_list
            self.ref_states_list=ref_states_list
            self.ref_bank_list=ref_bank_list
            self.ref_lengths_list=ref_lengths_list
            self.ref_prs_list = ref_prs_list
        else:
            ref_list=self.ref_list
            ref_states_list=self.ref_states_list
            ref_bank_list=self.ref_bank_list
            ref_lengths_list=self.ref_lengths_list
            ref_prs_list =self.ref_prs_list
            
  

        results = dict()
        results["predictions"] = [[] for _ in range(batch_size)]  # noqa: F812
        results["scores"] = [[] for _ in range(batch_size)]  # noqa: F812
        results["attention"] = [[] for _ in range(batch_size)]  # noqa: F812
        results["batch"] = batch
        if "tgt" in batch.__dict__:
            results["gold_score"] = self._score_target(
                batch, memory_bank, src_lengths, data, batch.src_map
                if data.data_type == 'text' and self.copy_attn else None)
            self.model.decoder.init_state(
                src, memory_bank, enc_states, with_cache=True)
        else:
            results["gold_score"] = [0] * batch_size

        # Tile states and memory beam_size times.
        self.model.decoder.map_state(
            lambda state, dim: tile(state, beam_size, dim=dim))
        if isinstance(memory_bank, tuple):
            memory_bank = tuple(tile(x, beam_size, dim=1) for x in memory_bank)

            mb_device = memory_bank[0].device
        else:
            memory_bank = tile(memory_bank, beam_size, dim=1)

            mb_device = memory_bank.device

        memory_lengths = tile(src_lengths, beam_size)
        src_map = (tile(batch.src_map, beam_size, dim=1)
                   if data.data_type == 'text' and self.copy_attn else None)

        if self.refer:
            for i in range(self.refer):
                self.extra_decoders[i].map_state(
                    lambda state, dim: tile(state, beam_size, dim=dim))
                if isinstance(ref_bank_list[i], tuple):
                    ref_bank_list[i] = tuple(tile(x, beam_size, dim=1) for x in ref_bank_list[i])
                else:
                    ref_bank_list[i] = tile(ref_bank_list[i], beam_size, dim=1)
                ref_lengths_list[i] = tile(ref_lengths_list[i], beam_size)
                ref_prs_list[i] = tile(ref_prs_list[i], beam_size).view(-1, 1)
            # ref_prs = torch.rand_like(ref_prs)
        else:
            ref_bank_list, ref_lengths_list = None, None

        top_beam_finished = torch.zeros([batch_size], dtype=torch.uint8)
        batch_offset = torch.arange(batch_size, dtype=torch.long)
        beam_offset = torch.arange(
            0,
            batch_size * beam_size,
            step=beam_size,
            dtype=torch.long,
            device=mb_device)
        alive_seq = torch.full(
            [batch_size * beam_size, 1],
            start_token,
            dtype=torch.long,
            device=mb_device)
        alive_attn = None

        # Give full probability to the first beam on the first step.
        topk_log_probs = (
            torch.tensor([0.0] + [float("-inf")] * (beam_size - 1),
                         device=mb_device).repeat(batch_size))

        # Structure that holds finished hypotheses.
        hypotheses = [[] for _ in range(batch_size)]  # noqa: F812

        for step in range(max_length):
            decoder_input = alive_seq[:, -1].view(1, -1, 1)

            log_probs, attn = \
                self._decode_and_generate(decoder_input, memory_bank,
                                          batch, data,
                                          memory_lengths=memory_lengths,
                                          src_map=src_map,
                                          step=step,
                                          batch_offset=batch_offset, ref_bank=ref_bank_list,
                                          ref_lengths=ref_lengths_list, ref_prs=ref_prs_list)

            vocab_size = log_probs.size(-1)

            if self.guide:
                log_probs = self.guide_by_tp(alive_seq, log_probs)

            if step < min_length:
                log_probs[:, end_token] = -1e20

            # Multiply probs by the beam probability.
            log_probs += topk_log_probs.view(-1).unsqueeze(1)

            alpha = self.global_scorer.alpha
            length_penalty = ((5.0 + (step + 1)) / 6.0) ** alpha

            # Flatten probs into a list of possibilities.
            curr_scores = log_probs / length_penalty
            curr_scores = curr_scores.reshape(-1, beam_size * vocab_size)
            topk_scores, topk_ids = curr_scores.topk(beam_size, dim=-1)

            # Recover log probs.
            topk_log_probs = topk_scores * length_penalty

            # Resolve beam origin and true word ids.
            topk_beam_index = topk_ids.div(vocab_size)
            topk_ids = topk_ids.fmod(vocab_size)

            # topk_ids = torch.cat([batch.tgt[step + 1][batch_offset].view(-1, 1), topk_ids], -1)[:, :self.beam_size]

            # Map beam_index to batch_index in the flat representation.
            batch_index = (
                    topk_beam_index
                    + beam_offset[:topk_beam_index.size(0)].unsqueeze(1))
            select_indices = batch_index.view(-1)

            # Append last prediction.
            alive_seq = torch.cat(
                [alive_seq.index_select(0, select_indices),
                 topk_ids.contiguous().view(-1, 1)], -1)
            if return_attention:
                current_attn = attn.index_select(1, select_indices)
                if alive_attn is None:
                    alive_attn = current_attn
                else:
                    alive_attn = alive_attn.index_select(1, select_indices)
                    alive_attn = torch.cat([alive_attn, current_attn], 0)

            is_finished = topk_ids.eq(end_token)
            if step + 1 == max_length:
                is_finished.fill_(1)

            # Save finished hypotheses.
            if is_finished.any():
                # Penalize beams that finished.
                topk_log_probs.masked_fill_(is_finished, -1e10)
                is_finished = is_finished.to('cpu')
                top_beam_finished |= is_finished[:, 0].eq(1)
                predictions = alive_seq.view(-1, beam_size, alive_seq.size(-1))
                attention = (
                    alive_attn.view(
                        alive_attn.size(0), -1, beam_size, alive_attn.size(-1))
                    if alive_attn is not None else None)
                non_finished_batch = []
                for i in range(is_finished.size(0)):
                    b = batch_offset[i]
                    finished_hyp = is_finished[i].nonzero().view(-1)
                    # Store finished hypotheses for this batch.
                    for j in finished_hyp:
                        hypotheses[b].append((
                            topk_scores[i, j],
                            predictions[i, j, 1:],  # Ignore start_token.
                            attention[:, i, j, :memory_lengths[i]]
                            if attention is not None else None))
                    # End condition is the top beam finished and we can return
                    # n_best hypotheses.
                    if top_beam_finished[i] and len(hypotheses[b]) >= n_best:
                        best_hyp = sorted(
                            hypotheses[b], key=lambda x: x[0], reverse=True)
                        for n, (score, pred, attn) in enumerate(best_hyp):
                            if n >= n_best:
                                break
                            results["scores"][b].append(score)
                            results["predictions"][b].append(pred)
                            results["attention"][b].append(
                                attn if attn is not None else [])
                    else:
                        non_finished_batch.append(i)
                non_finished = torch.tensor(non_finished_batch)
                # If all sentences are translated, no need to go further.
                if len(non_finished) == 0:
                    break
                # Remove finished batches for the next step.
                top_beam_finished = top_beam_finished.index_select(
                    0, non_finished)
                batch_offset = batch_offset.index_select(0, non_finished)
                non_finished = non_finished.to(topk_ids.device)
                topk_log_probs = topk_log_probs.index_select(0, non_finished)
                batch_index = batch_index.index_select(0, non_finished)
                select_indices = batch_index.view(-1)
                alive_seq = predictions.index_select(0, non_finished) \
                    .view(-1, alive_seq.size(-1))
                if alive_attn is not None:
                    alive_attn = attention.index_select(1, non_finished) \
                        .view(alive_attn.size(0),
                              -1, alive_attn.size(-1))

            # Reorder states.
            if isinstance(memory_bank, tuple):
                memory_bank = tuple(x.index_select(1, select_indices)
                                    for x in memory_bank)
            else:
                memory_bank = memory_bank.index_select(1, select_indices)

            memory_lengths = memory_lengths.index_select(0, select_indices)
            if self.refer:
                for i in range(self.refer):
                    if isinstance(ref_bank_list[i], tuple):
                        ref_bank_list[i] = tuple(x.index_select(1, select_indices)for x in ref_bank_list[i])
                    else:
                        ref_bank_list[i] = ref_bank_list[i].index_select(1, select_indices)

                    ref_lengths_list[i] = ref_lengths_list[i].index_select(0, select_indices)
                    ref_prs_list[i] = ref_prs_list[i].index_select(0, select_indices)
                    self.extra_decoders[i].map_state(
                        lambda state, dim: state.index_select(dim, select_indices))

            self.model.decoder.map_state(
                lambda state, dim: state.index_select(dim, select_indices))
            if src_map is not None:
                src_map = src_map.index_select(1, select_indices)
        if self.guide:
            print(self.batch_num)
            self.batch_num += 1
        return results

    def translate(self,
                  src_path=None,
                  src_data_iter=None,
                  src_length=None,
                  tgt_path=None,
                  tgt_data_iter=None,
                  src_dir=None,
                  batch_size=None,
                  attn_debug=False, search_mode=0, threshold=0,
                  ref_path=None):
        assert src_data_iter is not None or src_path is not None
        if batch_size is None:
            raise ValueError("batch_size must be set")
        data = inputters.build_dataset(self.fields,
                                       self.data_type,
                                       src_path=src_path,
                                       src_data_iter=src_data_iter,
                                       src_seq_length_trunc=src_length,
                                       tgt_path=tgt_path,
                                       tgt_data_iter=tgt_data_iter,
                                       src_dir=src_dir,
                                       sample_rate=self.sample_rate,
                                       window_size=self.window_size,
                                       window_stride=self.window_stride,
                                       window=self.window,
                                       use_filter_pred=self.use_filter_pred,
                                       ref_path=['%s.%d'%(ref_path, r) for r in range(self.refer)] if self.refer else None,
                                       ref_seq_length_trunc=self.max_sent_length,
									   ignore_unk=False)
        
        if self.cuda:
            cur_device = "cuda"
        else:
            cur_device = "cpu"
        if self.refer:
            for i in range(self.refer):
                data.fields['ref%d'%i].vocab = data.fields['src'].vocab

        data_iter = inputters.OrderedIterator(
            dataset=data, device=cur_device,
            batch_size=batch_size, train=False, sort=False,
            sort_within_batch=True, shuffle=False)

        if search_mode == 2:
            all_predictions = self.search(data_iter, data, src_path, train=False, threshold=threshold)
            for i in all_predictions:  #所有检索出来的相似代码
                self.out_file.write(i)   #
                self.out_file.flush()
            return

        builder = onmt.translate.TranslationBuilder(
            data, self.fields,
            self.n_best, self.replace_unk, tgt_path)

        # Statistics
        counter = count(1)
        pred_score_total, pred_words_total = 0, 0
        gold_score_total, gold_words_total = 0, 0

        all_scores = []
        all_predictions = []

        for batch in data_iter:
            batch_data = self.translate_batch(batch, data, fast=True, attn_debug=False)
            translations = builder.from_batch(batch_data)

            for trans in translations:
                all_scores += [trans.pred_scores[:self.n_best]]
                pred_score_total += trans.pred_scores[0]
                pred_words_total += len(trans.pred_sents[0])
                if tgt_path is not None:
                    gold_score_total += trans.gold_score
                    gold_words_total += len(trans.gold_sent) + 1

                n_best_preds = [" ".join(pred)
                                for pred in trans.pred_sents[:self.n_best]]
                all_predictions += [n_best_preds]
                # self.out_file.write('\n'.join(n_best_preds) + '\n')
                # self.out_file.flush()
        if search_mode == 1:
            sim_predictions = self.search(data_iter, data, src_path, threshold)
            for i in range(len(sim_predictions)):
                if not sim_predictions[i]:
                    self.out_file.write('\n'.join(all_predictions[i])+'\n')
                    self.out_file.flush()
                else:
                    self.out_file.write(sim_predictions[i])
                    self.out_file.flush()
        else:
            for i in all_predictions:
                self.out_file.write('\n'.join(i) + '\n')
                self.out_file.flush()
       # print(all_predictions)   #scores 是loss  all_predictions 是注释 [['translates an image on an image data into an image .']....]
        print(len(all_predictions))
        return all_scores, all_predictions

    def index_documents(self,
                        src_path=None,
                        src_data_iter=None,
                        tgt_path=None,
                        tgt_data_iter=None,
                        src_dir=None,
                        batch_size=None,
                        ):
        data = inputters.build_dataset(self.fields,
                                       self.data_type,
                                       src_path=src_path,
                                       src_data_iter=src_data_iter,
                                       src_seq_length_trunc=self.max_sent_length,
                                       tgt_path=tgt_path,
                                       tgt_data_iter=tgt_data_iter,
                                       src_dir=src_dir,
                                       sample_rate=self.sample_rate,
                                       window_size=self.window_size,
                                       window_stride=self.window_stride,
                                       window=self.window,
                                       use_filter_pred=self.use_filter_pred,
                                       ignore_unk=True)

        if self.cuda:
            cur_device = "cuda"
        else:
            cur_device = "cpu"

        data_iter = inputters.OrderedIterator(
            dataset=data, device=cur_device,
            batch_size=batch_size, train=False, sort=False,
            sort_within_batch=True, shuffle=False)

        doc_feats = []
        shard = 1
        for batch in data_iter:

            # Encoder forward.
            src = inputters.make_features(batch, 'src', data.data_type)
            _, src_lengths = batch.src
            enc_states, memory_bank, _ = self.model.encoder(src, src_lengths)
            feature = torch.max(memory_bank, 0)[0]
            _, recover_indices = torch.sort(batch.indices, descending=False)
            feature = feature[recover_indices]
            doc_feats.append(feature)
            if len(doc_feats) % 1250 == 0:
                print('saving shard %d' % shard)
                doc_feats = torch.cat(doc_feats)
                torch.save(doc_feats, '{}/indexes/codev{}.pt'.format('/'.join(src_path.split('/')[:2]), shard))

                doc_feats = []
                shard += 1
        if doc_feats:
            doc_feats = torch.cat(doc_feats)
            torch.save(doc_feats, '{}/indexes/codev{}.pt'.format('/'.join(src_path.split('/')[:2]), shard))
            print('done.')
    @staticmethod
    def load_indexes(src_path, shard):
        indexes = torch.load('{}/indexes/codev{}.pt'.format('/'.join(src_path.split('/')[:2]), shard))  # M*H

        return indexes

    def search(self, test_iter, data, src_path=None, threshold=0, train=False):  #src_path samples/%s/train/train.spl.src
        with open('{}/train/train.txt.tgt'.format('/'.join(src_path.split('/')[:2])), 'r') as tr:
            summaries = tr.readlines()
        with open('{}/train/train.spl.src'.format('/'.join(src_path.split('/')[:2])), 'r') as ts:
            sources = ts.readlines()
        
        all_summaries = []
        all_generated = []
        all_indexes = []
        for shard in range(1, 8):
            try:
                indexes = self.load_indexes(src_path, shard)
                all_indexes.append(indexes)
            except FileNotFoundError:
                pass
        all_indexes = torch.cat(all_indexes)
        for batch in test_iter:
            src = inputters.make_features(batch, 'src', data.data_type)
            _, src_lengths = batch.src
            last, memory_bank, _ = self.model.encoder(src, src_lengths)
            # props_v, props_idx = [], []

            props = self._search_batch(batch, memory_bank, all_indexes)
            if train:
                props = torch.topk(props, 6, dim=1)
                props_idx = props[1].tolist()
                props_v = props[0].tolist()
                # if random.random() > 0.4:
                #     generated = summaries[props_idx[1]]
                # else:
                for item, j in zip(props_idx, props_v):
                    generated = ' '.join([summaries[i].strip() for i in item[1:]])+'\n'
                    all_generated.append(generated)  #原来的注释

            else:
                props = torch.topk(props, 1, dim=1)  # B*2*k
                props_v = props[0][:, -1] #.append(props[0].unsqueeze(1))
                props_idx = props[1][:, -1] #.append((props[1]+40000*(shard-1)).unsqueeze(1))

                props_v = props_v.tolist()
                props_idx = props_idx.tolist()
                for item, j in zip(props_idx, props_v):
                    if j >= threshold:
                        generated = sources[item].strip()+'\n' #代码
                        all_generated.append(generated)
                        all_summaries.append(summaries[item].strip())
                    else:
                        all_generated.append('GENERATE\n')
                # self.out_file.write(generated)
                # self.out_file.flush()
        with open('{}/output/rnn.out'.format('/'.join(src_path.split('/')[:2])), 'w') as fwr:
            for s in all_summaries:
                fwr.write(s+'\n')
        return all_generated

    @staticmethod
    def _search_batch(batch, memory_bank, indexes):
        enc_states = torch.max(memory_bank, 0)[0]  # B*H
        _, recover_indices = torch.sort(batch.indices, descending=False)
        enc_states = enc_states[recover_indices]

        # props = CodeTranslator.pairwise_distances(enc_states, indexes)
        numerator = torch.mm(enc_states, indexes.transpose(0, 1))  # B*M
        denominator_1 = enc_states.norm(2, 1).unsqueeze(1)  # B*1
        denominator_2 = indexes.norm(2, 1).unsqueeze(1)  # M*1
        denominator = torch.mm(denominator_1, denominator_2.transpose(0, 1))  # B*M

        props = torch.div(numerator, denominator)  # B*M

        return props






