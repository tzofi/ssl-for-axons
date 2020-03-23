from tqdm import trange
import numpy as np
import itertools
from scipy.spatial.distance import cdist

""" Select permutations """
def select_permutations(num, classes, selection='max'):
    P_hat = np.array(list(itertools.permutations(list(range(num)), num)))
    n = P_hat.shape[0]
    
    for i in trange(classes):
        if i==0:
            j = np.random.randint(n)
            P = np.array(P_hat[j]).reshape([1,-1])
        else:
            P = np.concatenate([P,P_hat[j].reshape([1,-1])],axis=0)
        
        P_hat = np.delete(P_hat,j,axis=0)
        D = cdist(P,P_hat, metric='hamming').mean(axis=0).flatten()
        
        if selection=='max':
            j = D.argmax()
        else:
            m = int(D.shape[0]/2)
            S = D.argsort()
            j = S[np.random.randint(m-10,m+10)]

    return P

if __name__ == '__main__':
    p = select_permutations(8, 10)
    print(p)
