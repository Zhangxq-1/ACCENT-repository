import torch
import os
import time
import logging


# paths
dataset_dir = './dataset/java/'

if not os.path.exists(dataset_dir):
    raise Exception('Dataset directory not exist.')

train_source_path = os.path.join(dataset_dir, 'train/node.token')
train_code_path = os.path.join(dataset_dir, 'train/code.token')
train_nl_path = os.path.join(dataset_dir, 'train/javadoc.original')


valid_source_path = os.path.join(dataset_dir, 'dev/node.token')
valid_code_path = os.path.join(dataset_dir, 'dev/code.token')
valid_nl_path = os.path.join(dataset_dir, 'dev/javadoc.original')
valid_edage_path = os.path.join(dataset_dir, 'dev/adj.pkl')

test_source_path = os.path.join(dataset_dir, 'test/node.token')
test_code_path = os.path.join(dataset_dir, 'test/code.token')
test_nl_path = os.path.join(dataset_dir, 'test/javadoc.original')
test_edage_path = os.path.join(dataset_dir, 'test/adj.pkl')

model_dir = './model-adv/'
best_model_path = 'best_model.pt'

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

vocab_dir = './vocab-adv/'
source_vocab_path = 'same_vocab.pk'
code_vocab_path = 'code_vocab.pk'
nl_vocab_path = 'nl_vocab.pk'

source_vocab_txt_path = 'same_vocab.txt'
code_vocab_txt_path = 'code_vocab.txt'
nl_vocab_txt_path = 'nl_vocab.txt'

if not os.path.exists(vocab_dir):
    os.makedirs(vocab_dir)

out_dir = 'out/'    # other outputs dir
if not os.path.exists(out_dir):
    os.makedirs(out_dir)


# logger
log_dir = 'log/'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logger = logging.getLogger()
logger.setLevel(level=logging.INFO)
handler = logging.FileHandler(os.path.join(log_dir, time.strftime('%Y%m%d_%H%M%S', time.localtime())) + '.log')
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


# device
# use_cuda = False
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')


# features
trim_vocab_min_count = False
trim_vocab_max_size = True

use_coverage = False
use_pointer_gen = False
use_teacher_forcing = False
use_check_point = False
use_lr_decay = True
use_early_stopping = False

validate_during_train = True
save_valid_model = True
save_best_model = True
save_test_details = True


# limitations
max_code_length = 120
max_source_length = 200
max_nl_length = 50
min_nl_length = 3
max_decode_steps = 50
early_stopping_patience = 5


# hyperparameters
vocab_min_count = 1
source_vocab_size = 179621#123#50000
code_vocab_size = 19531#86007#19531 # 30000
nl_vocab_size = 26409#23314#26409    # 30000

embedding_dim = 256
hidden_size = 256
decoder_dropout_rate = 0.2
teacher_forcing_ratio = 0.5
batch_size = 8#32     # 128
source_encoder_lr = 0.001
code_encoder_lr = 0.001
ast_encoder_lr = 0.001
reduce_hidden_lr = 0.001
decoder_lr = 0.001
lr_decay_every = 1
lr_decay_rate = 0.99
n_epochs = 200    # 50

beam_width = 4
beam_top_sentences = 1     # number of sentences beam decoder decode for one input, must be 1 (eval.translate_indices)
eval_batch_size = 32    # 16
test_batch_size = 32

init_uniform_mag = 0.02
init_normal_std = 1e-4
eps = 1e-12


# visualization and resumes
print_every = 1000 # 1000
plot_every = 10000     # 100
save_model_every = 25000   # 2000
save_check_point_every = 10000   # 1000
validate_every = 10000     # 2000

# save config to log
save_config = True


#adv  config
lamta=0.2
lamta_decay=0.99
nearest_k_path=' '
var_everyCode_path=' '
embedd_path=' '
encoder_model_path=' '
embedd_src_path=' '
embedd_tgt_path=' '
original_code_path=' '


config_be_saved = ['dataset_dir', 'use_cuda', 'device', 'use_coverage', 'use_pointer_gen', 'use_teacher_forcing',
                   'use_lr_decay', 'use_early_stopping', 'max_code_length', 'max_nl_length', 'min_nl_length',
                   'max_decode_steps', 'early_stopping_patience']

train_config_be_saved = ['embedding_dim', 'hidden_size', 'decoder_dropout_rate', 'teacher_forcing_ratio',
                         'batch_size', 'source_encoder_lr', 'code_encoder_lr',
                         'decoder_lr', 'lr_decay_every', 'lr_decay_rate', 'n_epochs']

eval_config_be_saved = ['beam_width', 'beam_top_sentences', 'eval_batch_size', 'test_batch_size']

if save_config:
    config_dict = locals()
    logger.info('Configurations this run are shown below.')
    logger.info('Notes: If only runs test, the model configurations shown above is not ' +
                'the configurations of the model test runs on.')
    logger.info('')
    logger.info('Features and limitations:')
    for config in config_be_saved:
        logger.info('{}: {}'.format(config, config_dict[config]))
    logger.info('')
    logger.info('Train configurations:')
    for config in train_config_be_saved:
        logger.info('{}: {}'.format(config, config_dict[config]))
    logger.info('')
    logger.info('Eval and test configurations:')
    for config in eval_config_be_saved:
        logger.info('{}: {}'.format(config, config_dict[config]))
    logger.info('')
