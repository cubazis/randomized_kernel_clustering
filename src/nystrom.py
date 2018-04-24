import numpy as np
from scipy.linalg import hadamard

"""
pdist: Squared Euclidean distances between columns of A and B.
Thus, data points are assumed to be in columns, not rows
Dist = sqdist( A, B )
Dist = sqdist( A ) assumes B = A

Farhad Pourkamali-Anaraki, E-mail: Farhad.Pourkamali@colorado.edu
University of Colorado Boulder

Inputs: 
    - A: input matrix of size p*n1
    - B: input matrix of size p*n2 
Outputs: 
    - Dist: matrix of pairwise distances of size n1*n2
"""


def pdist(A, B=None):
    if (B is None):
        # Assume B = A
        AA = np.sum(A ** 2, 0)
        Dist = -2 * A.T.dot(A)
        Dist += AA[:, None] + AA
        print(Dist)
    else:
        assert A.shape == B.shape
        AA = np.sum(A ** 2, axis=0)
        BB = np.sum(B ** 2, axis=0)
        Dist = -2 * A.T.dot(B)
        Dist += AA[:, None] + BB
    return Dist

"""
findRep: Find a set of m landmark points Z=[z_1,...,z_m] for a given
dataset X=[x_1,...,x_n]

Farhad Pourkamali-Anaraki, E-mail: Farhad.Pourkamali@colorado.edu
University of Colorado Boulder
{
Inputs:
    - X: input data matrix of size pxn, where p is the dimension and n is
      the number of samples
    - m: number of desired landmark points (m<n)
    - param: a structure array in MATLAB with the specified field and values:
    - param.type: we consider various strategies to find the landmark set:
        1) 'uni-sample': uniformly sample m out of n data points
        2) 'kmeans-matlab': MATLAB implementation of K-means clustering 
        3) 'feature-extract-kmeans': randomized feature extraction
            algorithm for K-means clustering [Mahoney et al.]
    - param.dim: dimension of the reduced data for randomized K-means [Mahoney et al.]
    - param.iter: maximum number of iterations for iterative algorithms
                  (default is 10)
Outputs:
    - Z: landmark matrix of size pxm containing m representative points
}
"""


def findRep(X, m):
    p, n = X.shape
    indx = np.random.choice(n, m, replace=False)
    Z = X[:, list(indx)]
    return Z

def landmarks(X, m):
    n = X.shape[0]
    indx = np.random.choice(n, m, replace=False)
    Z = X[indx]
    return Z

"""
lowrank: хуйня ебаная
"""

def lowrank(X, Z, desiredRank, kernel, kernel_param):
    #assert X.shape[0] == Z.shape[0]
    #assert Z.shape[1] >= desiredRank
    if kernel == 'Poly':
        d = kernel_param[0]
        c = kernel_param[1]
        W = (Z.dot(Z.T) + c) ** d
        C = (X.dot(Z.T) + c) ** d
    else:
        raise ValueError()
    W = (W + W.T) / 2
    SW, UW = np.linalg.eig(W)
    I = SW.argsort()[::-1]
    SW = SW[I]
    UW = UW[:, I]
    SW = 1 / np.sqrt(SW[:desiredRank])
    UW = UW[:, :desiredRank] * SW
    L = C.dot(UW)
    print(L.shape)
    return L



'''

% if eigenvalue decomposition is desired (two outputs)
if nargout > 1
    U = E; 
    D = diag(D(1:desiredRank));
else     % if low-rank approximation is desired (one output)
    U = bsxfun(@times, E, sqrt(D(1:desiredRank))');
end

'''

def decom(X, Z, rank, kernel, kernel_param):
    #assert X.shape[0] == Z.shape[0]
    #assert Z.shape[1] >= rank
    if kernel == 'Poly':
        d = kernel_param[0]
        c = kernel_param[1]
        W = (Z.dot(Z.T) + c) ** d
        C = (X.dot(Z.T) + c) ** d
    W = (W + W.T) / 2
    Q, R = np.linalg.qr(C)
    M = (R * np.linalg.pinv(W) * R.T)
    M = (M + M.T) / 2
    D, V = np.linalg.eig(M)
    I = D.argsort()[::-1]
    D = D[I]
    V = V[I]
    E = Q.dot(V[:, :rank])
    U = E * np.sqrt(D[:rank])
    return U


''' https://octave-online.net/ '''

'''

A = [1 2; 3 4];

pkg load statistics

function [ Z ] = FindRep( X , m)
[p,n] = size(X);
indx = randsample(n,m);
Z = X(:,indx);
end

B = FindRep(A,1)

kernel.type  = 'Poly'; 
kernel.par   = [2,0];

function [U] = NysDecom(X , Z , desiredRank , kernel)
m = size(Z,2);
if size(X,1)~=size(Z,1), error('The given landmark set is not valid!'); end
if m < desiredRank, error('Select more landmark points!'); end

switch kernel.type
    case 'RBF'
        gamma = kernel.par;
        W = exp( -gamma.*sqdist(Z) );   % W: m*m
        C = exp( -gamma.*sqdist(X,Z) ); % C: n*m 
    case 'Poly'
        d = kernel.par(1); c = kernel.par(2); 
        W = (Z' * Z + c).^d; % W: m*m
        C = (X' * Z + c).^d; % C: n*m 
end

W = (W + W')/2; % make sure W is symmetric
[Q , R] = qr (C, 0);
M = (R * pinv(W) * R'); M = (M + M')/2;
[V,D] = eig(M);
[D,I] = sort(diag(D),'descend');
V = V(:, I);
E = Q * V(: , 1:desiredRank);
U = bsxfun(@times, E, sqrt(D(1:desiredRank))');

end

NysDecom(A, B, 1, kernel)

'''

def shit(X, Z, rank, kernel, kernel_param):
    #assert X.shape[0] == Z.shape[0]
    assert Z.shape[1] >= rank
    if kernel == 'Poly':
        d = kernel_param[0]
        c = kernel_param[1]
        W = (Z.dot(Z.T) + c) ** d
        C = (X.dot(Z.T) + c) ** d
    W = (W + W.T) / 2 # (R T HDK) T
    Q, R = np.linalg.qr(C) # тут должно быть W
    M = (R * np.linalg.pinv(W) * R.T)
    M = (M + M.T) / 2

    # тут надо решить B(Q T Ω) = (Q T W) чтобы найти B

    D, V = np.linalg.eig(M) # сюда пихаешь B а дальше все идентично
    I = D.argsort()[::-1]
    D = D[I]
    V = V[I]
    E = Q.dot(V[:, :rank])
    U = E * np.sqrt(D[:rank])
    return U

def nystroem(X, n_samples=20, r=2, kernel="polynomial", degree=2):
    inds = np.random.permutation(X.shape[0])
    basis_inds = inds[:n_samples]
    basis = X[basis_inds]

    if kernel == "polynomial":
        C = (X @ basis.T) ** degree
        W = (basis @ basis.T) ** degree

    V, S, _ = np.linalg.svd(W)
    V = V[:,:r]
    S_inverse = np.sqrt(1/S)[:r]

    L = C @ V * S_inverse

    K = L @ L.T

    uu, ss, _ = np.linalg.svd(L.T @ L)
    Y = L @ uu * np.sqrt(ss)

    return K, Y

def efficient(X, r=2, l=6):
    r_l = r + l
    n = X.shape[0]

    need_size, bin_pow = closest_two_power_my(n)
    diff = need_size - n

    sample = np.random.permutation(n)[:r_l]
    R = np.identity(n)[:, sample]


    R = np.concatenate([R, np.zeros((diff, r_l))])
    RT = R.T

    K = (X @ X.T) ** 2
    K_pad = np.concatenate([K, np.zeros((diff,n))])
    K_pad = np.concatenate([K_pad, np.zeros((need_size,diff))],axis=1)

    D = np.random.randint(2, size=need_size)*2-1
    D = np.diag(D)

    h = hadamard(need_size)
    g1 = D @ K_pad
    g2 = h @ (g1)
    W = RT @ g2

    W = W.T


    u,s,v = np.linalg.svd(W)
    q = u[:,:r_l]

    SIGMA = D @ (h @ R)

    left = SIGMA.T @ q
    right = W.T @ q
    B = np.linalg.inv(left) @ right

    u, s, v = np.linalg.svd(B.T)
    u = u[:,:r]
    s = s[:r]
    y = ((np.sqrt(s) * u).T @ q.T).T[:X.shape[0]]

    K = y @ y.T

    return K, y

def closest_two_power_my(n):
    count = 0

    if (n and not(n&(n-1))):
        return n

    while( n != 0):
        n >>= 1
        count += 1
    res = 1 << count
    return res, count

def closest_two_power(n):
    ceil_pow = np.ceil(np.log2(n))
    return int(2 ** ceil_pow), ceil_pow