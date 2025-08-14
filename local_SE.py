import numpy as np
import numpy.linalg as LA
import scipy.sparse.linalg as sLA
import scipy.sparse as sp
import networkx as nx
import scipy

# 1-D RBF kernel
def RBF(x, y, omega):
    return np.exp(-((x-y)**2)/(2*omega**2))


def sparse_SE(A, d, which='LA'):
    # A must be a sparse matrix
    if which=='LA':
        w, v = sLA.eigsh(A, k=d, which='LA')
        idx = np.argsort(w)[::-1]
    elif which=='LM':
        w, v = sLA.eigsh(A, k=d, which='LM')
        idx = np.argsort(abs(w))[::-1]
    w = w[idx]
    v = v[:,idx]
    X_hat = v[:,0:d] @ np.diag(abs(w[0:d])**(1/2))
    return np.real(X_hat)


def SE(A, d, which='LA'):
    w, v = LA.eig(A)
    if which=='LA':
        idx = np.argsort(w)[::-1]
    elif which=='LM':
        idx = np.argsort(abs(w))[::-1]
    w = w[idx]
    v = v[:,idx]
    X_hat = v[:,0:d] @ np.diag(abs(w[0:d])**(1/2))
    return np.real(X_hat)


def local_embedding(A, d, weights, eigs_by='LA'):
    W_sr = sp.diags_array((weights)**(0.5)).astype('f')
    sym_w_A = W_sr @ A @ W_sr
    X_sym = sparse_SE(sym_w_A, d, which=eigs_by)
    X_new = np.diag(weights**(-0.5)) @ X_sym
    return X_new


def local_eigenvalues_gd(A, z, local_strength, m=6):
    
    n = A.shape[0]
    G = nx.from_numpy_matrix(A)
    D = dict(nx.shortest_path_length(G))
    
    eigvals = np.zeros((len(local_strength), m))
    weights = np.zeros(n)
    weights[np.array(list(D[z].keys()))] = 1 / (np.array(list(D[z].values())) + 1)

    
    for j in range(len(local_strength)):
        W = scipy.sparse.diags( (weights**(local_strength[j]) / sum(weights**(local_strength[j])))**(1/2) )
        w, v = np.real(sLA.eigsh(W@A@W, m, which='LA'))
        w = w[::-1]
        eigvals[j,:] = w
        
    return(eigvals)

def weighting(Z, z, tau):
    weights = np.exp( - tau*(Z-Z[z])**2)
    return weights / np.mean(weights)

def graph_dist_weighting(A, z):
    n = A.shape[0]
    G = nx.from_numpy_matrix(A)
    D = dict(nx.shortest_path_length(G))
    weights = np.zeros(n)
    weights[np.array(list(D[z].keys()))] = 1 / (np.array(list(D[z].values())) + 1)
    return weights / np.mean(weights)

        
    