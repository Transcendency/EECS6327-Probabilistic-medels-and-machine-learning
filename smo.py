import sys
import random

import numpy as np
from math import sqrt
from scipy.spatial.distance import squareform, cdist

import cPickle

random.seed(1024 * 1024)

##################################################################################################################
'''
SMO
'''

def load_dataset():
    f = open('../data/mnist.pkl', 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    return train_set, valid_set, test_set

def select_sample():
    train_set, _, test_set = load_dataset()
    three = [train_set[0][i] for i in range(len(train_set[1])) if train_set[1][i] == 3]
    five = [train_set[0][i] for i in range(len(train_set[1])) if train_set[1][i] == 5]
    three_t = [test_set[0][i] for i in range(len(test_set[1])) if test_set[1][i] == 3]
    five_t = [test_set[0][i] for i in range(len(test_set[1])) if test_set[1][i] == 5]
    return three,five,three_t,five_t

def Gaussian_kernel(X_i, X_j, sigma = 0.01):
    pairwise_dists = cdist(X_i, X_j, 'euclidean')
    K = np.exp(-(pairwise_dists ** 2) * sigma)
    return np.matrix(K)

def linear_kernel(Xi, Xj):
    return Xi * Xj.T

def map_label(Y):
    m = {k:v for v,k in enumerate(set(Y))}
    Y_map = [0] * len(Y)
    for i, k in enumerate(Y):
        Y_map[i] = m[k]
    return Y_map

def randomizeX(X1,X2):
    X = X1 + X2
    Y = np.append(np.ones(len(X1),dtype = np.int8),(-np.ones(len(X2),dtype = np.int8)))
    t = zip(X,Y)
    random.shuffle(t)
    X,Y = zip(*t)
    return np.matrix(list(X)),np.matrix(list(Y)).reshape((-1,1))

class SMO:

    def __init__(self,kernel_type):
        self.model = None
        self.kernel_type = kernel_type

    def train(self, X, Y, C = 0.5, tol = 1e-4, max_passes = 5):
        K = Gaussian_kernel
        if self.kernel_type == "linear_kernel": 
            K = linear_kernel

        m, n = X.shape
        alphas = np.matrix(np.zeros([m, 1]))
        b = 0.0
       
        K_cache = K(X, X)

        print >> sys.stderr, 'Done with K_cache with shape', K_cache.shape
 
        iter = 0
        passes = 0
        
        while passes < max_passes:
            iter += 1
            
            if iter % 10 == 0: sys.stderr.write('.')
            if iter % 500 == 0: sys.stderr.write('%d Iters\n' % iter)

            # print >> sys.stderr, 'Iter :', iter
            num_changed_alphas = 0
            for i in range(m):
                fx_i = alphas.T * np.multiply(Y, K_cache[:, i]) + b
                y_i = Y[i]
                E_i = fx_i - y_i
                alpha_i = alpha_ii = alphas[i]

                if (y_i * E_i < -tol and alpha_i < C) or (y_i * E_i > tol and alpha_i > 0.0):
                    while True:
                        j = random.randint(0, m - 1)
                        if i != j: break
                    
                    fx_j = alphas.T * np.multiply(Y, K_cache[:, j]) + b
                    y_j = Y[j]
                    E_j = fx_j - y_j
                    alpha_j = alpha_jj = alphas[j]
                    if y_i != y_j:
                        L = max(0.0, alpha_j - alpha_i)
                        H = min(C, C + alpha_j - alpha_i)
                    else:
                        L = max(0.0, alpha_i + alpha_j - C)
                        H = min(C, alpha_i + alpha_j)
                    if L == H: continue
                
                    eta = 2 * K_cache[i, j] - K_cache[i, i] - K_cache[j, j]
                    if eta >= 0.0: continue
                    
                    alpha_j = alpha_j - (y_j * (E_i - E_j) / eta)
                    if alpha_j > H: alpha_j = H
                    if alpha_j < L: alpha_j = L
                    if abs(alpha_jj - alpha_j) < tol: continue
                    
                    alpha_i = alpha_i + (y_i * y_j * (alpha_jj - alpha_j))
                    
                    b_i = b - E_i - y_i * (alpha_i - alpha_ii) * K_cache[i, i] - y_j * (alpha_j - alpha_jj) * K_cache[i, j]
                    b_j = b - E_j - y_i * (alpha_i - alpha_ii) * K_cache[i, j] - y_j * (alpha_j - alpha_jj) * K_cache[j, j]
                    
                    if alpha_i > 0.0 and alpha_i < C:
                        b = b_i
                    elif alpha_j > 0.0 and alpha_j < C:
                        b = b_j
                    else:
                        b = (b_i + b_j) / 2.0
                    
                    alphas[i] = alpha_i
                    alphas[j] = alpha_j
                    
                    num_changed_alphas = num_changed_alphas + 1

            if num_changed_alphas == 0:
                passes += 1
            else:
                passes = 0
                        
        sys.stderr.write('\n classifer' + str(id) + ' Done training with Iter %d\n' % iter)

        self.model = dict()
        alpha_index = [ index for index, alpha in enumerate(alphas) if alpha > 0.0 ]
        
        self.model['X'] = X[alpha_index]
        self.model['Y'] = Y[alpha_index]
        self.model['kernel'] = K
        self.model['alphas'] = alphas[alpha_index]
        self.model['b'] = b
        self.model['w'] = X.T * np.multiply(alphas, Y)  
   
    def predict(self, X_test):
        m, n = X_test.shape
        fx = np.matrix(np.zeros([m, 1]))
        if self.model['kernel'] == linear_kernel:
            w = self.model['w']
            b = self.model['b']
            fx = X_test * w + b
        else:
            alphas = self.model['alphas']
            X = self.model['X']
            Y = self.model['Y']
            K = self.model['kernel']
            b = self.model['b']
            fx = np.multiply(np.tile(Y, [m]), K(X, X_test)).T * alphas + b
        return fx
    
    def test(self, X, Y):
        fx = self.predict(X)
        Y_pred = np.matrix(np.zeros(Y.shape))
        Y_pred[np.where(fx >= 0)] = 1
        Y_pred[np.where(fx < 0)] = -1

        P = np.matrix(np.zeros(Y.shape))
        P[np.where(Y_pred == Y)] = 1 

        return 1.0 * P.sum() / len(Y)

if __name__ == "__main__":
    three, five, three_t, five_t = select_sample()
    X_train, Y_train = randomizeX(three,five)
    X_test, Y_test = randomizeX(three_t,five_t)

    clf = SMO(kernel_type = "linear_kernel")
    clf.train(X_train, Y_train)

    acc_train = clf.test(X_train, Y_train)
    acc_test  = clf.test(X_test, Y_test)

    print >> sys.stderr, 'Training accuracy for SMO with Gaussian kernel : %lf%%' % (100 *  acc_train)
    print >> sys.stderr, 'Test accuracy for SMO with Gaussian kernel : %lf%%' % (100 *  acc_test)
