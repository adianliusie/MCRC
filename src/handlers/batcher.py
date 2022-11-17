import torch
import random

from typing import List, Tuple
from types import SimpleNamespace

class Batcher():
    """ Class that takes care of all the data batching to be used in training"""
    def __init__(self, max_len:int=512, device:str='cuda'):
        self.max_len = max_len
        self.device  = device

    #== Main batching method ======================================================================#
    def batches(self, data:list, bsz:int, shuffle:bool=False):
        """splits the data into batches and returns them"""
        examples = self._prep_examples(data)
        if shuffle: random.shuffle(examples)
        batches = [examples[i:i+bsz] for i in range(0,len(examples), bsz)]
        for batch in batches:
            yield self.batchify(batch)
     
    def _prep_examples(self, data:list):
        """ sequence classification input data preparation"""
        prepped_examples = []
        for ex in data:
            ex_id = ex.ex_id
            label = ex.answer
            input_ids = ex.input_ids
            
            # if ids larger than max size, then truncate
            max_opt_len = max([len(opt_ids) for opt_ids in ex.input_ids])
            if self.max_len and (max_opt_len>self.max_len): 
                input_ids = [[opt_ids[0]] + opt_ids[-self.max_len+1:] for opt_ids in ex.input_ids]
            
            prepped_examples.append([ex_id, input_ids, label])
        return prepped_examples

    def batchify(self, batch:List[list]):
        """each input is input ids and mask for utt, + label"""
        ex_id, input_ids, labels = zip(*batch)  
        input_ids, attention_mask = self._get_3D_padded_ids(input_ids)
        labels = torch.LongTensor(labels).to(self.device)
        return SimpleNamespace(ex_id=ex_id, 
                               input_ids=input_ids, 
                               attention_mask=attention_mask, 
                               labels=labels)

    #== Util Methods ==============================================================================#
    def to(self, device:torch.device):
        """ sets the device of the batcher """
        self.device = device
    
    def _get_padded_ids(self, ids:list, pad_id:int=0)->Tuple[torch.LongTensor]:
        """ pads 2D input ids arry so that every row has the same length """
        max_len = max([len(x) for x in ids])
        padded_ids = [x     + [pad_id]*(max_len-len(x)) for x in ids]
        mask       = [[1]*len(x) + [0]*(max_len-len(x)) for x in ids]
        ids = torch.LongTensor(padded_ids).to(self.device)
        mask = torch.FloatTensor(mask).to(self.device)
        return ids, mask

    def _get_3D_padded_ids(self, ids:list, pad_id:int=0)->Tuple[torch.LongTensor]:
        max_len = max([len(x) for row in ids for x in row])
        padded_ids = [[x     + [pad_id]*(max_len-len(x)) for x in row] for row in ids]
        mask       = [[[1]*len(x) + [0]*(max_len-len(x)) for x in row] for row in ids]
        ids = torch.LongTensor(padded_ids).to(self.device)
        mask = torch.FloatTensor(mask).to(self.device)
        return ids, mask
                           
    def __call__(self, data, bsz, shuffle=False):
        """routes the main method do the batches function"""
        return self.batches(data=data, bsz=bsz, shuffle=shuffle)

    