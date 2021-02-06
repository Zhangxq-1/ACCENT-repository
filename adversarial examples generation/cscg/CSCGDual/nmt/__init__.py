#from util import read_corpus, data_iter, read_corpus_for_dsl, get_new_batch, data_iter_for_dual
from CSCGDual.nmt.util import *
import CSCGDual.vocab
from CSCGDual.nmt.model import NMT, to_input_variable
from CSCGDual.nmt.nmt import get_bleu, decode
