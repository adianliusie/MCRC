import torch
import torch.nn.functional as F

from captum.attr import Saliency, LayerIntegratedGradients, IntegratedGradients
from captum.attr import visualization as viz
from copy import deepcopy

from .evaluater import Evaluator

class InterpretableLoader(Evaluator):
    def saliency(self, data_name:str=None, mode:str='test', idx:int=0, ex=None, visible=True):
        #load example to run interpretability on
        assert (ex is None) or (data_name is None)
        if data_name:
            ex = self.get_ex(data_name=data_name, mode=mode, idx=idx) 

        #get prediction info
        logits = self.model(input_ids=ex.input_ids).logits
        pred = torch.argmax(logits).item()

        #get input tokens
        tokens = self.tokenizer.convert_ids_to_tokens(ex.input_ids[0][pred].tolist())
        
        #do saliency
        inputs_embeds = self.model.electra.embeddings.word_embeddings(ex.input_ids)

        saliency = Saliency(self.model_forward_embeds)
        attributions = saliency.attribute(inputs=inputs_embeds,
                                          target=pred, 
                                          abs=False, 
                                          additional_forward_args=(ex.attention_mask)
        )

        #attributions over entire input embeddings- sum over each word and only select the row of the predicted option
        attributions = torch.sum(attributions[0], dim=-1)[pred]
        attributions = attributions - torch.mean(attributions)
        attributions = attributions / torch.norm(attributions)

        #visualise
        if visible:
            prob = F.softmax(logits, dim=-1)[0, pred]
            vis = viz.VisualizationDataRecord(
                    attributions,
                    prob,
                    pred,
                    str(ex.labels[0].item()),
                    pred,
                    attributions.sum(),       
                    tokens,
                    1)
            viz.visualize_text([vis])

        return tokens, attributions 

    def integrad(self, data_name:str=None, mode:str='test', idx:int=0, ex=None, visible=True):
        #load example to run interpretability on
        assert (ex is None) or (data_name is None)
        if data_name:
            ex = self.get_ex(data_name=data_name, mode=mode, idx=idx) 
            
        #prepare inputs
        inputs_embeds = self.model.electra.embeddings.word_embeddings(ex.input_ids)
        baseline_embeds = self.baseline_embeds(ex)

        #get relevant input info
        logits = self.model(input_ids=ex.input_ids).logits
        pred = torch.argmax(logits).item()

        #get input tokens
        tokens = self.tokenizer.convert_ids_to_tokens(ex.input_ids[0, pred].tolist())
        
        #set up the ingtegrated gradients
        lig = IntegratedGradients(self.model_forward_embeds)

        attributions, delta = lig.attribute(inputs=inputs_embeds,
                                            baselines=baseline_embeds,
                                            return_convergence_delta=True, 
                                            internal_batch_size=8,
                                            target=pred, 
                                            additional_forward_args=(ex.attention_mask),
        )
        
        #attributions over entire input embeddings- sum over each word and only select the row of the predicted option
        print(attributions.shape)
        attributions = torch.sum(attributions[0], dim=-1)[pred]

        print(attributions.shape, attributions.sum())
        attributions = attributions / torch.norm(attributions)

        #visualise
        if visible:
            prob = F.softmax(logits, dim=-1)[0, pred]
            vis = viz.VisualizationDataRecord(
                    attributions,
                    prob,
                    pred,
                    str(ex.labels[0].item()),
                    pred,
                    attributions.sum(),       
                    tokens,
                    1)
            viz.visualize_text([vis])

        return tokens, attributions             
        
    def model_forward(self, input_ids:torch.LongTensor, attention_mask:torch.LongTensor):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return output.logits
    
    def model_forward_embeds(self, inputs_embeds:torch.LongTensor, attention_mask:torch.LongTensor):
        output = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        return output.logits

    def baseline_ids(self, ex):
        # creating the baseline ids
        baseline_ids = deepcopy(ex.input_ids)
        #baseline_ids[:, 0] = self.tokenizer.cls_token_id
        #baseline_ids[:, -1] = self.tokenizer.sep_token_id
        #baseline_ids[:, 1:-1] = self.tokenizer.pad_token_id 
        baseline_ids[:, :] = self.tokenizer.pad_token_id 

        return baseline_ids

    def baseline_embeds(self, ex):
        baseline_ids = self.baseline_ids(ex)
        baseline_embeds = self.model.electra.embeddings.word_embeddings(baseline_ids)
        return baseline_embeds
    
    def get_ex(self, data_name:str, mode:str, idx:int):
        data_set = self.data_handler.prep_split(data_name=data_name, mode=mode)
        data_set = list(self.batcher(data_set, bsz=1))
        ex = data_set[idx]
        return ex

    @property
    def tokenizer(self): 
        return self.data_handler.tokenizer
