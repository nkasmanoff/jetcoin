"""
Misclelaneous helper functions. For now just the ones used for allowing balanced sampling.
"""

import numpy as np
import matplotlib.pyplot as plt 

def get_bins(params, num_bins=10):
    sample_counts, bin_edges, _ = plt.hist(params,num_bins)
    return sample_counts, bin_edges

def get_bin_num(bin_edges,value):
    # Given a data point, return the number bin it is in!
    bin_edges = np.asarray(bin_edges)
    idx = (np.abs(bin_edges - value.item())).argmin()
    if value > bin_edges[idx]:
        return idx
    else:
        if idx > 0:
            return idx - 1
        else:
            return idx
