import numpy as np
from scipy.special import comb

i = 0

def compute_ari(table):
    """
    Compute ari, given the index table
    :param table: (r, s)
    :return:
    """
    
    # (r,)
    a = table.sum(axis=1)
    # (s,)
    b = table.sum(axis=0)
    n = a.sum()
    
    comb_a = comb(a, 2).sum()
    comb_b = comb(b, 2).sum()
    comb_n = comb(n, 2)
    comb_table = comb(table, 2).sum()
    
    if (comb_b == comb_a == comb_n == comb_table):
        # the perfect case
        ari = 1.0
    else:
        ari = (
            (comb_table - comb_a * comb_b / comb_n) /
            (0.5 * (comb_a + comb_b) - (comb_a * comb_b) / comb_n)
        )
    
    return ari
    
    
def compute_mask_ari(mask0, mask1):
    """
    Given two sets of masks, compute ari
    :param mask0: ground truth mask, (N0, H, W)
    :param mask1: predicted mask, (N1, H, W)
    :return:
    """
    
    # will first need to compute a table of shape (N0, N1)
    # (N0, 1, H, W)
    mask0 = mask0[:, None].byte()
    # (1, N1, H, W)
    mask1 = mask1[None, :].byte()
    # (N0, N1, H, W)
    agree = mask0 & mask1
    # (N0, N1)
    table = agree.sum(dim=-1).sum(dim=-1)
    
    return compute_ari(table.numpy())

if __name__ == '__main__':
    table = np.array([
        [3, 0, 1],
        [1, 2, 1],
        [0, 2, 2],
    ])
    
    print(compute_ari(table))
