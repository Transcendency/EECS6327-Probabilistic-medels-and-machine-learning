import numpy as np
import cPickle, gzip
import math


# Load the dataset
def load_dataset():
    f = open('../data/mnist.pkl', 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    return train_set, valid_set, test_set


## calculate loss
def sq_loss(hx, y):
    loss = np.transpose(hx - y).dot(hx - y) / 2 * len(hx - y)
    return loss

def correct_ratio(hx, y):
    a = np.sum(np.isclose(hx, y,atol=0.5))
    ratio = a*1.0 / len(y)
    return ratio

def compute_accuracy(w):
    '''
    input: `w' the parameters computed by noraml equaiton
    return: the accuracy predicted by `w'
    '''
    _, _, test = load_dataset()
    tx,ty = test[0],test[1]
    m = len(ty)
    it = np.ones(shape=(m, 1))
    ntx = np.concatenate((it, tx), axis=1)
    hx = ntx.dot(w)
    result = np.argmax(hx,axis = 1)
    print "Solution computed by normal equaiton " + str(correct_ratio(result,ty))

def Compute_parameters():
    train, valid, test = load_dataset()
    x,y = train[0],train[1]
    # number of training samples
    m = len(y)
    # add a column of ones to X (interception data)
    it = np.ones(shape=(m, 1))
    nx = np.concatenate((it, x), axis=1)
    oneHoty = np.zeros((len(y),10))
    for u in range(len(oneHoty)):
        oneHoty[u][y[u]] =1
    w = (np.linalg.pinv(np.transpose(nx).dot(nx))).dot(np.transpose(nx)).dot(oneHoty)
    return w

if __name__ == '__main__':
    compute_accuracy(Compute_parameters())