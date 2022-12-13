import torch
import pickle
import numpy as np
import os

from tqdm import tqdm 
import torch.nn.functional as F


from .trainer import Trainer
from ..data.handler import DataHandler
from ..loss.cross_entropy import CrossEntropyLoss
from ..utils.analysis import anneal_probs

class Evaluator(Trainer):
    """ Evaluator class- inherits Trainer so has all experiment methods
        class takes care of evaluation and automatic caching of results"""

    def __init__(self, path, device='cuda'):
        self.exp_path = path
        self.device = device

    def setup_helpers(self):
        args = self.load_args('model_args.json')
        super().setup_helpers(args)
        self.load_model()
        self.model_loss = CrossEntropyLoss(self.model)

    #== Model Prediction Methods ==================================================================#
    def load_preds(self, dataset:str, mode:str='test', formatting:str=None)->dict:
        probs = self.load_probs(dataset, mode, formatting)
        preds = {}
        for ex_id, probs in probs.items():
            preds[ex_id] = int(np.argmax(probs, axis=-1))  
        return preds
        
    def load_probs(self, dataset:str, mode:str='test', formatting=None, calibrate=False)->dict:
        """ loads cached probabilities, if not cached then generate """
        if not self.probs_exist(dataset, mode, formatting):
            self.setup_helpers()
            if formatting: self.data_handler.formatting = formatting
            probs = self.generate_probs(dataset, mode)
            self.cache_probs(probs, dataset, mode, formatting)
        probs = self.load_cached_probs(dataset, mode, formatting)

        if calibrate:
            labels = self.load_labels(dataset, mode)
            probs = anneal_probs(probs, labels)
        return probs

    @torch.no_grad()
    def generate_probs(self, dataset:str, mode:str='test'):
        """ get model probabilities for each example in dataset"""
        self.model.eval()
        self.to(self.device)
        eval_data = self.data_handler.prep_split(dataset, mode)
        eval_batches = self.batcher(
            data = eval_data, 
            bsz = 1, 
            shuffle = False
        )        
        probs = {}
        
        for batch in tqdm(eval_batches):
            ex_id = batch.ex_id[0]
            output = self.model_loss(batch)

            logits = output.logits.squeeze(0)
            if logits.shape and logits.shape[-1] > 1:  # Get probabilities of predictions
                prob = F.softmax(logits, dim=-1)
            probs[ex_id] = prob.cpu().numpy()
        return probs

    #== loading and saving functions ==============================================================#
    def get_eval_path(self, dataset:str, mode:str, formatting:str=None):
        if formatting is None: 
            args = self.load_args('model_args.json')
            formatting = args.formatting
        eval_name = f'{dataset}_{mode}_{formatting}'
        pred_path = os.path.join(self.exp_path, 'eval', f'{eval_name}.pk')
        return pred_path
    
    def cache_probs(self, probs, dataset:str, mode:str, formatting:str=None):
        pred_path = self.get_eval_path(dataset, mode, formatting)
        with open(pred_path, 'wb') as handle:
            pickle.dump(probs, handle)
    
    def load_cached_probs(self, dataset:str, mode:str, formatting:str=None):
        pred_path = self.get_eval_path(dataset, mode, formatting)
        with open(pred_path, 'rb') as handle:
            probs = pickle.load(handle)
        return probs
    
    def probs_exist(self, dataset:str, mode:str, formatting:str):
        pred_path = self.get_eval_path(dataset, mode, formatting)
        return os.path.isfile(pred_path)

    #== general eval methods ======================================================================#
    @staticmethod
    def load_labels(dataset:str, mode:str='test', lim=None)->dict:
        eval_data = DataHandler.load_split(dataset, mode)
        labels_dict = {}
        for ex in eval_data:
            labels_dict[ex.ex_id] = ex.label
        return labels_dict

    @staticmethod
    def load_split(dataset:str, mode:str='test', lim=None)->dict:
        eval_data = DataHandler.load_split(dataset, mode)
        output_dict = {}
        for ex in eval_data:
            output_dict[ex.ex_id] = ex
        return output_dict

    @staticmethod
    def calc_acc(preds, labels):
        assert preds.keys() == labels.keys(), "keys don't match"
        hits = sum([preds[idx] == labels[idx] for idx in labels.keys()])
        acc = hits/len(preds)
        return 100*acc

