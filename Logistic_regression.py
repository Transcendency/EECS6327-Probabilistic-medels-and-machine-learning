import numpy as np
import cPickle, gzip
import math
import argparse
import sys
import matplotlib.pyplot as plt

# Load the dataset
def load_dataset():
    f = open('../data/mnist.pkl', 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    return train_set, valid_set, test_set

## loss for logistic regression
def cross_entropy(hx,y):
    loss = -np.log(hx[range(hx.shape[0]),y])
    loss = np.sum(loss)
    return loss / hx.shape[0]

## compute correct ratio
def correct_ratio(hx,y):
    ratio = np.sum(np.equal(hx,y))*1.0 / len(y)
    return ratio

## compute scores for output layer
def softmax(hx):
    hp = np.exp(hx)
    denom =  1.0 / np.sum(hp,axis=1,keepdims=True)
    for i in range(len(denom)):
        hp[i:i+1][:] = hp[i:i+1][:] * denom[i:i+1]
    return hp

def augument_x(x,y):
    m = len(y)
    it = np.ones(shape=(m,1))
    return np.concatenate((it,x),axis=1)

def statistic(x,y,w):
    hx = softmax(np.matmul(augument_x(x,y),w))
    cost = cross_entropy(hx,y)
    indices = hx.argmax(axis=1)
    cr = correct_ratio(indices,y)
    return cost, indices, cr

def plot(tc,tec,ert,erte,lr,epochs):
    plt.plot(tc,color = "red",label = "training cost")
    plt.plot(tec,color = "blue", label = "test_cost")
    plt.title('COST CURVE WITH SIZE ' + " learning rate " + str(lr) + " epochs " + str(epochs) )
    plt.legend(loc='upper left')
    plt.savefig("../figure/0.01cost.png")
    plt.show()

    plt.plot(ert, color = "green", label ="training_accuracy")
    plt.plot(erte, color = "yellow", label = "evaluation_accuracy")
    plt.title('ERROR RATE CURVE WITH SIZE ' + " learning rate " + str(lr) + " epochs " + str(epochs) )
    plt.legend(loc='upper left')
    plt.savefig("../figure/0.01acc.png")
    plt.show()  


def logstic_regression(alpha = 0.1,epochs=10):
    train, valid, test = load_dataset()
    x = train[0]
    y = train[1]
    test_x = test[0]
    test_y = test[1]
    valid_x = valid[0]
    valid_y = valid[1]

    numClass = 10

    m = len(y)
    nx = augument_x(x,y)
    w = np.random.uniform(0.1,1,size=(len(nx[0]),numClass))
    cost = 0
    last_epoch_cost = 0
    
    ert,erte,tc,tec = [],[],[],[]

    ##  GRADIENT DECENT
    for i in range(epochs): 
        hx = softmax(np.matmul(nx,w))
        yOneHot = np.zeros_like(hx)
        yOneHot[np.arange(len(y)),y] = 1
        
        if cost < cross_entropy(hx,y):
            alpha = alpha * 0.9
            
        for u in range(len(w)):
            for l in range(len(w[u])):
                w[u][l] = w[u][l] - alpha*(1.0/len(hx)) * (np.dot(np.transpose(nx)[u],np.transpose(hx-yOneHot)[l]))

        cost = cross_entropy(hx,y)
        ## STATISTIC ON TRAINING DATA
        train_cost, train_indices, cr = statistic(x,y,w)
        tc.append(train_cost)
        ert.append(1-cr)
        print str(i) + "th Epoch, "+ "train "+ str(train_indices[:5]) + ' ' +str(y[:5]) + " train cost " + str(train_cost) + " train error ratio " + str(1 - cr)

        ## STATISTIC ON TEST DATA
        test_cost, test_indices, cr = statistic(test_x,test_y,w)
        tec.append(test_cost)
        erte.append(1-cr)
        print str(i) + "th Epoch, "+ "test " + str(test_indices[:5]) + ' ' +str(test_y[:5]) + " test cost " + str(test_cost) + " test error ratio " + str(1 - cr)
        print

    valid_cost, valid_indices, cr = statistic(valid_x,valid_y,w)
    print "VALID " + str(valid_indices[:5]) + ' ' +str(valid_y[:5]) + " valid cost " + str(valid_cost) + " valid error ratio " + str(1 - cr)  + " with lr" + str(alpha)

    plot(tc,tec,ert,erte,alpha,epochs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', help='learning for the logstic_regression',type=float)
    parser.add_argument('--epochs', help='learning for the logstic_regression',type=int)
    args = parser.parse_args()
    if not args.alpha or not args.epochs: 
        parser.print_help()
        sys.exit(1)
    else:
        logstic_regression(alpha = args.alpha,epochs = args.epochs)