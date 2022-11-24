import numpy as np
import matplotlib.pyplot as plt

from typing import List, Dict
from scipy import special, stats

#== probs conversion functions ====================================================================#
def probs_to_logits(probs:dict)->dict:
    output = {}
    for idx, prob in probs.items():
        logits = np.log(prob)
        output[idx] = logits
    return output

def probs_to_entropies(probs:dict)->dict:
    output = {}
    for idx, ex_prob in probs.items():
        entropy = _calc_entropy(ex_prob)
        output[idx] = entropy
    return output

def _calc_entropy(prob_np:np.ndarray):
    return stats.entropy(prob_np, base=2, axis=-1)

def anneal_probs(probs:dict)->dict:
    pass

#== numpy helper functions ========================================================================#
def convert_dict_np(input_dict:Dict[str, np.ndarray])->np.ndarray:
    output = []
    for k, p in sorted(input_dict.items()):
        output.append(p)
    output = np.array(output)
    return output

def probs_to_preds_np(probs_np:np.ndarray)->np.ndarray:
    return np.argmax(probs_np, axis=1)

def get_hits_np(preds_np:np.ndarray, labels_np:np.ndarray):
    return preds_np == labels_np

def calc_entropy_np(probs_np:np.ndarray)->np.ndarray:
    return stats.entropy(probs_np, base=2, axis=1)

def anneal_probs_np(probs_np:np.ndarray, labels_np:np.ndarray):
    #get current model probs and avg max prob
    logits = np.log(probs_np)
    max_probs_np = [max(i) for i in probs_np]
    avg_prob = np.mean(max_probs_np)
    
    #look at current model accuracy
    preds = probs_to_preds_np(probs_np)
    hits = get_hits_np(preds, labels_np)
    acc = np.mean(hits)
    
    #do the annealing
    a = 1
    while avg_prob > acc:  
        a += 0.001
        annealed_logits = logits/a
        probs_np  = special.softmax(annealed_logits, axis=1)
        max_probs = [max(i) for i in probs_np]
        avg_prob = np.mean(max_probs)
    print(avg_prob, acc)
    return probs_np

#== plotting functions ============================================================================#
def entropy_plot(probs:dict, labels:dict, ax1=None, ax2=None, color='blue', calibrate=False):
    """ get effective number of options distribution """
    
    #create twin axis if not provided
    if ax1 is None and ax2 is None:
        fig, ax1 = plt.subplots(figsize=(12,8))
        ax2 = ax1.twinx()
        plt.ylim(0)

    # convert dicts to numpy array
    probs = convert_dict_np(probs)
    labels = convert_dict_np(labels)
    preds = probs_to_preds_np(probs)
    hits = get_hits_np(preds, labels)
   
    # calculate entropy and effective number of options
    if calibrate: probs = anneal_probs_np(probs, labels)
    entropies = calc_entropy_np(probs)
    eff_num_opt = 2**entropies

    # get effective number of options for correctly answered questions
    probs_c = [prob for prob, hit in zip(probs, hits) if hit==1] 
    entropies_c = calc_entropy_np(probs_c)
    eff_num_opt_c = 2**entropies_c
    
    # binning points and calculating accuracies 
    bins = [i/100 for i in range(100,401, 20)]
    hist, bin_edges = np.histogram(eff_num_opt, bins=bins)
    hist_c, _ = np.histogram(eff_num_opt_c, bins=bins)

    # plotting
    bin_centres = np.array(bins[:-1]) + 0.1
    accuracies  = np.array(hist_c)/np.array(hist)

    ax1.plot(bin_centres, hist, marker='.', color=color, linewidth=4, markersize=18)
    ax2.plot(bin_centres, accuracies, marker='.', linestyle=(0, (3, 1, 1, 1)), color=color, linewidth=4, markersize=18)
    
#== probs conversion functions ====================================================================#
def probs_to_distractor_entropies(probs:dict, labels:dict)->dict:
    """ gets entropies at the distractor level"""
    assert probs.keys() == labels.keys()

    logits = probs_to_logits(probs)

    output = {}
    for idx in sorted(labels.keys()):
        ex_logits = logits[idx]
        answer = labels[idx]

        for k in range(len(ex_logits)):
            # skip correct answer
            if k == answer: continue    
            dist_prob = special.softmax([ex_logits[answer], ex_logits[k]])
            output[(idx,k)] = stats.entropy(dist_prob)

    return output