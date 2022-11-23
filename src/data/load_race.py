import os
import pickle

from datasets import load_dataset
from types import SimpleNamespace

from ..utils.general import save_pickle, load_pickle, load_json
from .download import RACE_C_DIR, download_race_plus_plus

def load_race(levels=['M', 'H', 'C']):
    #load RACE-M and RACE-H data from hugginface
    race_data = {}
    if 'M' in levels: race_data['M'] = load_dataset("race", "middle")
    if 'H' in levels: race_data['H'] = load_dataset("race", "high")
    if 'C' in levels: race_data['C'] = load_race_c()

    #load and format each split, for each difficulty level, and add to data
    SPLITS = ['train', 'validation', 'test']
    train_all, dev_all, test_all = [], [], []
    for key, data in race_data.items():
        train, dev, test = [format_race(data[split], key) for split in SPLITS]
        train_all += train
        dev_all   += dev
        test_all  += test

    return train_all, dev_all, test_all

def format_race(data, char):
    """ converts dict to SimpleNamespace for QA data"""
    outputs = []
    ans_to_id = {'A':0, 'B':1, 'C':2, 'D':3}
    for k, ex in enumerate(data):
        ex_id = f'{char}_{k}'
        question = ex['question']
        context  = ex['article']
        options  = ex['options']
        answer   = ans_to_id[ex['answer']]
        ex_obj = SimpleNamespace(ex_id=ex_id, 
                                 question=question, 
                                 context=context, 
                                 options=options, 
                                 label=answer)
        outputs.append(ex_obj)
    return outputs

#== Loading for RACE-C ============================================================================#
def load_race_c():    
    # Download data if missing
    if not os.path.isdir(RACE_C_DIR):
        download_race_plus_plus()
    
    # Load cached data if exists, otherwise process and cache
    pickle_path = os.path.join(RACE_C_DIR, 'cache.pkl')    
    if os.path.isfile(pickle_path):
        train, dev, test = load_pickle(pickle_path)
    else:
        splits_path = [f'{RACE_C_DIR}/{split}' for split in ['train', 'dev', 'test']]
        train, dev, test = [load_race_c_split(path) for path in splits_path]
        save_pickle(data=[train, dev, test], path=pickle_path)
        
    return {'train':train, 'validation':dev, 'test':test}

def load_race_c_split(split_path:str):
    file_paths = [f'{split_path}/{f_path}' for f_path in os.listdir(split_path)]
    outputs = []
    for file_path in file_paths:
        outputs += load_race_file(file_path)
    return outputs

def load_race_file(path:str):
    file_data = load_json(path)
    article = file_data['article']
    answers = file_data['answers']
    options = file_data['options']
    questions = file_data['questions']
    
    outputs = []
    assert len(questions) == len(options) == len(answers)
    for k in range(len(questions)):
        ex = {'question':questions[k], 
              'article':article,
              'options':options[k],
              'answer':answers[k]}
        outputs.append(ex)
    return outputs
