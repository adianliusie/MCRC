import random
from copy import deepcopy

from .load_race import load_race
from .load_reclor import load_reclor
#from ..handlers.ensemble_evaluater import EnsembleEvaluator
from ..utils.analysis import probs_to_entropies

def make_adv_race(levels=['M', 'H', 'C']):
    train_race, dev_race, test_race = load_race(levels)

#== data pruning probe dataset ====================================================================#
""" def load_pruned_race():
    #get shortcut model predictions, and convert to entropies
    system_path = '/home/al826/rds/hpc-work/2022/QA/MCMRC/trained_models/electra-large-QO'
    system = EnsembleEvaluator(system_path)
    probs = system.load_probs('race', 'train')
    entropies = probs_to_entropies(probs)
    sorted_idx, _ = sorted(entropies.items(), key=lambda x: x[1], reverse=True)

    train, dev, test = load_race() """


#== random context probing datasets ===============================================================#
def load_random_context_race(levels=['M', 'H', 'C']):
    train, dev, test = load_race(levels)
    train = shuffle_context(train)
    dev = shuffle_context(dev)
    test = shuffle_context(test)
    return train, dev, test

def load_random_context_reclor():
    train, dev, test = load_reclor()
    train = shuffle_context(train)
    dev = shuffle_context(dev)
    test = shuffle_context(test)
    return train, dev, test

def shuffle_context(data):
    #get all contexts in dataset and shuffle
    contexts = [ex.context for ex in data]
    random.seed(1)
    random.shuffle(contexts)

    # change contexts for each sample
    data = deepcopy(data)
    for k, ex in enumerate(data):
        ex.context = contexts[k]
    return data