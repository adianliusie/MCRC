import torch
import torch.nn.functional as F
import numpy as np

from typing import List
from types import SimpleNamespace
from copy import deepcopy
from collections import defaultdict

from transformers.models.electra.modeling_electra import ElectraSelfAttention

from .electra_patched import ImportanceSelfAttention
from ...handlers.evaluater import Evaluator

class WordImportanceWeights(torch.nn.Module):
    """ Parameter class for differentiable Search"""
    def __init__(self, seq_len:int, sigmoid_w:float=1, log_w:float=100):
        super().__init__()
        self.weights = torch.nn.Parameter((0.00001/(log_w*sigmoid_w))*torch.randn(seq_len))
        self.sigmoid_w = sigmoid_w
        self.log_w = log_w

    @property
    def importance(self):
        alphas = torch.sigmoid(self.sigmoid_w * self.weights)
        norm_alphas = alphas/torch.max(alphas)
        #alphas = torch.abs(self.weights)
        #norm_alphas = torch.max(alphas) - alphas
        #norm_alphas = F.relu(1 - norm_alphas)
        return norm_alphas

    @property
    def importance_scores(self):
        scores = self.log_w * torch.log(self.importance)
        scores = scores.unsqueeze(0).unsqueeze(0).unsqueeze(0) #formatting for transformer
        return scores

    def get_percentiles(self, percentiles:List[int]):
        importance = self.importance.detach().cpu().numpy()
        stats = [np.percentile(importance, i) for i in percentiles]

        importance_scores = self.importance_scores.detach().cpu().numpy()
        scores_stats = [np.percentile(importance_scores, i) for i in percentiles]
        return stats + scores_stats

    def get_importance_order(self):
        importance = self.importance.detach().cpu().numpy()
        return np.argsort(importance)

class ImportanceAnalyser(Evaluator):
    def __init__(self, path, device='cuda'):
        # load project
        self.exp_path = path
        super().setup_helpers()
        self.model.eval()
        self.to(device)
        self.device = device

    #== General util methods ======================================================================#
    def get_ex(self, dataset:str, mode:str, k:int=0)->SimpleNamespace:
        data = self.data_handler.prep_split(dataset, mode, lim=10) #TEMP
        ex = data[k]
        return ex

    def prepare_ex(self, ex:SimpleNamespace):
        ex = self.data_handler._prep_ids([ex])
        batch = next(self.batcher(ex, bsz=1))
        return batch

    def model_forward(self, batch):
        output = self.model(
            input_ids = batch.input_ids, 
            attention_mask = batch.attention_mask, 
            labels=batch.labels,
            output_hidden_states=True
        )

        h = self.model.sequence_summary(output.hidden_states[-1])

        return SimpleNamespace(
            logits=output.logits,
            h = h
        )

    #== Brute force search method ================================================================#

    def brute_force_search(self, ex:SimpleNamespace, opt_num=None, eps=0.02):
        if opt_num:
            ex = deepcopy(ex)
            ex.options = [ex.options[opt_num]]
            ex.label = 0

        # prepare input
        batch = self.prepare_ex(ex)

        # get original hidden representation
        output = self.model_forward(batch)
        h_start = output.h.detach()
        logit_start = output.logits

        # variables for the search
        del_pos = set()
        del_attn_mask = deepcopy(batch.attention_mask)
        dist = 0

        # do the brute search
        while dist < eps:
            position_scores = defaultdict(lambda x: 0)

            for k in range(batch.input_ids.size(-1)):
                if k in del_pos: continue
                
                # prepare mask to mask current position
                mask = del_attn_mask.clone()
                mask[:,:,k] = 0
                batch.attention_mask = mask

                # get score after masking current position
                output = self.model_forward(batch)
                dist = torch.mean(torch.abs(output.h-h_start))
                position_scores[k] = [dist.item(), output.logits.item()]

            # select the position that cuases smallest change in distance
            k_min, (dist, logit) = min(position_scores.items(), key=lambda x: x[1][0])
            print(k_min, dist, logit_start, logit)
            del_attn_mask[:,:,k_min] = 0
            del_pos.add(k_min)

        print(self.data_handler.tokenizer.decode(batch.input_ids[0][0]), '\n\n')
        del_attn_mask = del_attn_mask.bool()
        survived_input_ids = batch.input_ids[del_attn_mask]
        print(self.data_handler.tokenizer.decode(survived_input_ids))
        return del_pos

    def differentiable_search(self, ex:SimpleNamespace, opt_num=None, lr=1e-3, sigmoid_w:float=1, log_w:float=1):
        # prepare input
        batch = self.prepare_ex(ex)

        #get original hidden representation
        start_output = self.model_forward(batch)
        h_start = start_output.h.detach()
        print('starting logits ', readable(start_output.logits[0]))

        # initialise word importance container
        self.word_weights = WordImportanceWeights(
            seq_len = batch.input_ids.size(-1),
            sigmoid_w = sigmoid_w,
            log_w = log_w
        )
        optimizer = torch.optim.AdamW(
            self.word_weights.parameters(), 
            lr=lr
        )

        #change the model set up
        self.set_model_differentiable_mask(self.word_weights)
        
        # do search for best word importance weights
        loss_logger = np.zeros(2)
        logit_logger = np.zeros(4)
        for k in range(1,10_001):
            output = self.model_forward(batch)
            h = output.h

            # calculate loss
            if opt_num:
                sim_loss = 10*torch.mean(torch.abs(h[opt_num]-h_start[opt_num]))
                #sim_loss = -1 * output.logits[0][opt_num]
            else:
                sim_loss = 10*torch.mean(torch.abs(h-h_start))
            
            L1_loss = torch.mean(self.word_weights.importance)
            loss = sim_loss + L1_loss

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # logging
            loss_logger += [sim_loss.item(), L1_loss.item()]
            logit_logger += output.logits[0].detach().cpu().numpy()

            #printing
            if k%50==0:
                percentiles = self.word_weights.get_percentiles([10,50,90])
                print(k, readable(loss_logger/50), readable(percentiles, 3), readable(logit_logger/50, 2))
                loss_logger = np.zeros(2)
                logit_logger = np.zeros(4)
        return self.word_weights

    #== Utils to allow for differentiable search ==================================================#
    def set_model_differentiable_mask(self, word_weights:WordImportanceWeights):
        word_weights.to(self.device)
        for layer in self.model.electra.encoder.layer:
            layer.attention.self.__class__ = ImportanceSelfAttention
            layer.attention.self.importance_init(word_weights)

    def reset_model(self):
        for layer in self.model.electra.encoder.layer:
            layer.attention.self.__class__ = ElectraSelfAttention
    
    def reduce_input(self, ex, word_weights=None, eps=0.02):
        if word_weights is None:
            word_weights = self.word_weights

        # prepare input
        batch = self.prepare_ex(ex)

        self.reset_model()
        for k in word_weights.get_importance_order():
            batch.attention_mask[:,:,k] = 0
            output = self.model_forward(batch)

            L1_dist = torch.mean(torch.abs(output.h-h_start))
            logit = output.logits

            print(L1_dist, logit)
            if L1_dist > eps:
                break
    
            

def readable(x:list, p:int=4):
    if torch.is_tensor(x):
        x = x.cpu().tolist() 
    return [round(i, p) for i in x]