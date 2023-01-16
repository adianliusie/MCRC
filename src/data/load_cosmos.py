import random

from copy import deepcopy
from types import SimpleNamespace
from datasets import load_dataset
from typing import List, Tuple

def load_cosmos():
    #load RACE-M and RACE-H data from hugginface
    dataset = load_dataset("cosmos_qa")

    train_data = list(dataset['train'])
    train, dev = _create_splits(train_data, 0.8)
    test       = list(dataset['validation'])

    train = format_cosmos(train)
    dev = format_cosmos(dev)
    test = format_cosmos(test)
    
    return train, dev, test

def format_cosmos(data:List[dict]):
    outputs = []
    for ex in data:
        ex_id    = ex['id']
        question = ex['question']
        context  = ex['context']
        options  = [ex[f'answer{k}'] for k in [0,1,2,3]]
        answer   = ex['label']
        ex_obj = SimpleNamespace(ex_id=ex_id, 
                                 question=question, 
                                 context=context, 
                                 options=options, 
                                 label=answer)
        outputs.append(ex_obj)
    return outputs

def _create_splits(examples:list, ratio=0.8)->Tuple[list, list]:
    examples = deepcopy(examples)
    split_len = int(ratio*len(examples))
    
    random.seed(1)
    random.shuffle(examples)
    
    split_1 = examples[:split_len]
    split_2 = examples[split_len:]
    return split_1, split_2
