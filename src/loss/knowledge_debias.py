import torch
import torch.nn.functional as F

from types import SimpleNamespace
from typing import Tuple

from .base import BaseLoss
from .utils import get_entropy

class KnowledgeDebiasLoss(BaseLoss):
    def __init__(self, model, args):
        super().__init__()
        self.model = model
        self.alpha = args.entropy_alpha

    def forward(self, batch: SimpleNamespace) -> Tuple[float, dict]:
        QOC_output = self.model(
            input_ids = batch.QOC_ids, 
            attention_mask = batch.QOC_mask, 
            labels=batch.labels
        )

        QO_output = self.model(
            input_ids = batch.QO_ids, 
            attention_mask = batch.QO_mask, 
        )

        # Baseline Model predictions
        QOC_logits = QOC_output.logits
        QOC_loss = QOC_output.loss

        # Shortcut Model predictions
        QO_logits = QO_output.logits
        QO_entropy = torch.mean(get_entropy(QO_logits))
        
        loss = QOC_loss - self.alpha * QO_entropy

        # Get accuracy of the various systems
        QOC_hits = torch.argmax(QOC_logits, dim=-1) == batch.labels
        QO_hits  = torch.argmax(QO_logits,  dim=-1) == batch.labels

        QOC_acc = QOC_hits.sum()/len(batch.labels)
        QO_acc = QO_hits.sum()/len(batch.labels)

        #record training metrics
        self.record_metrics({
            'loss': loss.item(),
            'ce':QOC_loss,
            'acc': QOC_acc.item(),
            'QO_entropy':QO_entropy,
            'QO_acc': QO_acc.item(),
        })

        return SimpleNamespace(
                    loss=loss, 
                    logits=QOC_logits,
                    QO_logits=QO_logits
        )


