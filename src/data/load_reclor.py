import os
from types import SimpleNamespace

from .download import RECLOR_DIR, download_reclor
from ..utils.general import save_pickle, load_pickle, load_json

def load_reclor():    
    # download data if missing
    if not os.path.isdir(RECLOR_DIR):
        download_reclor()
    
    # load and prepare each data split
    splits_path = [f'{RECLOR_DIR}/{split}.json' for split in ['train', 'val', 'val']]
    train, dev, test = [load_reclor_split(path) for path in splits_path]
    return train, dev, test

def load_reclor_split(split_path:str):
    split_data = load_json(split_path)
    outputs = []
    for ex in split_data:
        ex_id    = ex['id_string']
        question = ex['question']
        context  = ex['context']
        options  = ex['answers']
        answer   = ex['label']
        ex_obj = SimpleNamespace(ex_id=ex_id, 
                                 question=question, 
                                 context=context, 
                                 options=options, 
                                 label=answer)
        outputs.append(ex_obj)
    return outputs
