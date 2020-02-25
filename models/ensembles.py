import numpy as np

def linear_combination_betas(n,round_to = 10):
    betas = np.random.rand(n)
    return np.round(betas/sum(betas),round_to)

    