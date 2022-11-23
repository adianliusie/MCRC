import random

from typing import List
from types import SimpleNamespace
from tqdm import tqdm
from copy import deepcopy
from functools import lru_cache

from ..models.tokenizers import load_tokenizer
from .load_race import load_race

#== Main DataHandler class ========================================================================#
class DataHandler:
    def __init__(self, trans_name:str, formatting:str='QOC'):
        self.tokenizer = load_tokenizer(trans_name)
        self.formatting = formatting
        self.knowledge_debias = False

    #== MCRC Data processing (i.e. tokenizing text) ===============================================#
    def prep_split(self, data_name:str, mode:str, lim=None):
        data = self.load_split(data_name, mode, lim)
        return self._prep_ids(data)
    
    def prep_data(self, data_name, lim=None):
        train, dev, test = self.load_data(data_name=data_name, lim=lim)
        train, dev, test = [self._prep_ids(split) for split in [train, dev, test]]
        return train, dev, test
    
    def _prep_ids(self, split_data):
        split_data = deepcopy(split_data)
        for ex in tqdm(split_data):
            Q_ids   = self.tokenizer(ex.question).input_ids
            C_ids   = self.tokenizer(ex.context).input_ids
            options = [self.tokenizer(option).input_ids for option in ex.options]

            if self.knowledge_debias:
                ex.QOC_ids = self._prep_inputs(Q_ids, C_ids, options, 'QOC')
                ex.QO_ids = self._prep_inputs(Q_ids, C_ids, options, 'QO')
            else:
                ex.input_ids = self._prep_inputs(Q_ids, C_ids, options, self.formatting)

        return split_data
            
    def _prep_inputs(self, Q_ids:List[int], C_ids:List[int], options:List[List[int]], formatting):
        if formatting == 'QOC':
            ids = [C_ids + Q_ids[1:-1] + O_ids[1:] for O_ids in options]
        elif formatting == 'O':
            ids = [O_ids for O_ids in options]
        elif formatting == 'QO':
            ids = [Q_ids[:-1] + O_ids[1:] for O_ids in options]
        elif formatting == 'CO':
            ids = [C_ids + O_ids[1:] for O_ids in options]
        return ids
    
    #== Data loading utils ========================================================================#
    @classmethod
    def load_split(cls, data_name:str, mode:str, lim=None):
        split_index = {'train':0, 'dev':1, 'test':2}
        data = cls.load_data(data_name, lim)[split_index[mode]]
        return data
    
    @classmethod
    @lru_cache(maxsize=5)
    def load_data(cls, data_name:str, lim=None):
        if   data_name == 'race++': train, dev, test = load_race(levels=['M', 'H', 'C'])
        elif data_name == 'race':   train, dev, test = load_race(levels=['M', 'H'])
        elif data_name == 'reclor': train, dev, test = None, None, None #load_reclor()
        elif data_name == 'cosmos': train, dev, test = None, None, None #load_cosmos()
        else: raise ValueError(f"{data_name}: invalid dataset name") 
        if lim:
            train = rand_select(train, lim)
            dev   = rand_select(dev, lim)
            test  = rand_select(test, lim)
        return train, dev, test
    
#== Misc utils functions ============================================================================#
def rand_select(data:list, lim:None):
    if data is None: return None
    random_seed = random.Random(1)
    data = data.copy()
    random_seed.shuffle(data)
    return data[:lim]
