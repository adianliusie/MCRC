import torch

def get_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Computes the entropy based on final dimension."""
    lsoftmax = torch.log_softmax(logits, dim = -1)
    entropy = -torch.exp(lsoftmax) * lsoftmax
    entropy = torch.sum(entropy, dim=-1)
    return entropy
