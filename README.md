# Adversarial Robustness of Deep Code Comment Generation<br>
Official implementation of our paper.<br>
## requirements

Ubuntu 20.04

tqdm<br>
nltk<br>
prettytable<br>
torch>=1.3.0<br>
scikit-learn:0.22.1<br>
gensim:3.8.0<br>
Javalang:0.11.0<br>
numpy:1.18.5<br>
pandas:1.0.5<br>
## DataSet
The dataset used in the experiment is in data dir(contains java and python dataset), which you can download and put into the folder.


## Train and Test
This part will show how to generate adversarial examples and how to conduct the masked training.

### pre-training

0. First of all, for adversarial example generation, you need to train the original model first. You can refer to the oirginal work for detailed details(I will introduce GNN separately):

[Transormer and Lstm model](https://github.com/wasiahmad/NeuralCodeSum)       and      [CSCGDual model](https://github.com/Bolin0215/CSCGDual)

#### GNN
cd gnn

config.py is the configuration file and you can should the path in this file.

run python3 main.py to train GNN model.


### adversarial examples generation

The adversarial example generation dir contains the four different models to generate adversarial examples.

#### Transformer and lstm model 

1.create a data dir and put the train, dev and test data into it.

2.training word embedding and  encoder

  (1)cd encoder
  
  (2)run python3 vocab_embedding.py  to train word embedding.
  
  (3)run python train.py
  
note: Don't forget to change the path, or you may meet an error!

3.generate adversarial exmaple

  (1)run python3 extract_formalparater.py
  
    ---extract_formalparater.py needs "code.original" as the input, you need to change the path.
  
  (2)run python3 extract_vaname.py
    
    ---extract_varname.py needs "/code.original" as the input, you need to change the path.
    
    ---the output of this file are two part: candidate identifier dataset extract from the train data and identifier for every code snippet in test dataset.
    
    --for extract the candidate identifier dataset, you need put the train data as "/code.original", and for identifier for every code snippet in test dataset,you need put the test as "/code.original".

  
  (3)run python3 near_k_voc.py
  
    ---near_k_voc.py needs: "code.original",   formalparater from 3-(1), candidate identifier dataset from 3-(2) and identifier for each code snippet from 3-(2). you need to put this this file.
    
    --the output is the nearest k identifer for each identifier in each code snippet.
  
  (4)python3 substitution.py 
  
  ---substitution.py needs "code.original", "javadoc.original" in the dataset, word embedding from 2-(1), the nearest k identifer for each identifier in each code snippet from 3-(3), pretrained model form 0. 
  
  and finally, run python3 fool.py to generate adversarial exmaples.

note: for CSCGDual and GNN, the steps are the same, you just need to change the corresponding path.

### masked training

The relevant files are all in the masking training directory.

1. extract identifier for each code snippet from 3-(2) for the train dataset.
2. change this "  var_everyCode_path='../var_for_everyCode.pkl' (in adversary.py)"  file with generated file in the previous step.
3. train the masked model as same process as the pre-training  stag.

