
attack_root='../rencos/'
lang= 'java'

model =[attack_root+'models/java/baseline_spl_step_100000.pt']

src=attack_root+'samples/'+lang+'/test/test.spl.src'
output = attack_root+'samples/'+lang+'/output/test.out'

refer= 2
beam =1
batch_size = 1
gpu= 0 
fast = True

max_sent_length= 300
min_length=3 
max_length = 30

ref_path=attack_root+'samples/'+lang+'/test/test.ref.src'


word2vec_path=attack_root+'embedding/train/node_w2v_64'

original_code_path=attack_root+'samples/java/test/test.txt.src'
summary_path=attack_root+'samples/java/test/test.txt.tgt'

nearest_k_path=attack_root+'var_name/java/test/java_test_nearest_top5.pkl'
var_everyCode_path=attack_root+'var_name/java/test/var_for_everyCode.pkl'



encoder_word2vec_src_path=attack_root+'encoder/vocab/train/node_w2v_code_64'
encoder_word2vec_tgt_path=attack_root+'encoder/vocab/train/node_w2v_summ_64'
encoder_model_file=attack_root+'encoder/model.pkl'

saved_path=attack_root+'var_name/java/test/test_java_IR_data.pkl'


len_sent=300  # j and p :300

codeLen=300
summLen=50  
