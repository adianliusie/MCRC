import torch

from .cross_entropy import CrossEntropyLoss
from .knowledge_debias import KnowledgeDebiasLoss

def get_loss(loss:str, model:torch.nn.Module, args=None):
    if loss == 'cross-entropy':
        return CrossEntropyLoss(model)
    elif loss == 'knowledge-debias':
        return KnowledgeDebiasLoss(model, args)
