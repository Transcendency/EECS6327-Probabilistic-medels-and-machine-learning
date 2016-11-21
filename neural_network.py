import random
import cPickle, gzip
import sys
import matplotlib.pyplot as plt
import ast
import argparse
import numpy as np

class CrossEntropyCost(object):

    @staticmethod
    def fn(a, y):
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    @staticmethod
    def delta(z, a, y):
        return (a-y)


#### Main Network class
class NN(object):

    def __init__(self, sizes, cost=CrossEntropyCost):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.weight_initializer()
        self.cost=cost

    def weight_initializer(self):
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def feedforward(self, a):
        a = np.reshape(a,(-1,1))
        for b, w in zip(self.biases[:-1], self.weights[:-1]):
            a = sigmoid(np.dot(w, a)+b)
        lz = np.dot(self.weights[-1],a) + self.biases[-1]
        lac = softmax(lz)
        return lac

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            lmbda = 0.0,
            evaluation_data=None,
            m_evaluation_cost=True,
            m_evaluation_accuracy=True,
            m_training_cost=True,
            m_training_accuracy=True):

        if evaluation_data: n_data = len(evaluation_data)
        n = len(training_data)
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(
                    mini_batch, eta, lmbda, len(training_data))
            print "Epoch %s training complete" % j
            if m_training_cost:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                print "Cost on training data: {}".format(cost)
            if m_training_accuracy:
                accuracy = self.accuracy(training_data, convert=False)
                training_accuracy.append(accuracy)
                print "Accuracy on training data: {} / {}".format(
                    accuracy, n)
            if m_evaluation_cost:
                cost = self.total_cost(evaluation_data, lmbda, convert=False)
                evaluation_cost.append(cost)
                print "Cost on evaluation data: {}".format(cost)
            if m_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy)
                print "Accuracy on evaluation data: {} / {}".format(
                    self.accuracy(evaluation_data), n_data)
            print
        return evaluation_cost, evaluation_accuracy, \
            training_cost, training_accuracy

    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        nb = [np.zeros(b.shape) for b in self.biases]
        nw = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            dnb, dnw = self.backprop(x, y)
            nb = [nb+dnb for nb, dnb in zip(nb, dnb)]
            nw = [nw+dnw for nw, dnw in zip(nw, dnw)]
        self.weights = [(1-eta*(lmbda/n))*w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nw)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nb)]



    def backprop(self, x, y):
        nb = [np.zeros(b.shape) for b in self.biases]
        nw = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = np.reshape(x,(-1,1))
        activations = [activation]
        excitation = []
        for b, w in zip(self.biases[:-1], self.weights[:-1]):
            z = np.dot(w, activation)+b
            excitation.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        lz = np.dot(self.weights[-1],activation) + self.biases[-1]
        excitation.append(lz)
        lac = softmax(lz)
        activations.append(lac)
        # backward pass
        delta = (self.cost).delta(excitation[-1], activations[-1], y)
        nb[-1] = delta
        nw[-1] = np.dot(delta, activations[-2].transpose())

        for l in xrange(2, self.num_layers):
            z = excitation[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nb[-l] = delta
            nw[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nb, nw)

    def accuracy(self, data, convert=False):
        if convert:
            results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                       for (x, y) in data]
        else:
            results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in data]
        return sum(int( 1 == y[x]) for (x, y) in results)

    def total_cost(self, data, lmbda, convert=False):
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            if convert: y = vectorized_result(y)
            cost += self.cost.fn(a, y)/len(data)
        cost += 0.5*(lmbda/len(data))*sum(
            np.linalg.norm(w)**2 for w in self.weights)
        return cost

def vectorize_labels(j):
    l = np.zeros((len(j),10,1))
    l[np.arange(len(j)),j] = 1
    return l

def softmax(hx):
    """
    compute probability.
    """
    hp = np.exp(hx)
    denom =  1.0 / np.sum(hp)
    for i in range(len(hp)):
        hp[i][0] = hp[i][0] * denom
    return hp

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

def augumentX(x):
    ax = [np.append(d,[1]) for d in x]
    return ax

def load_dataset():
    f = open('../data/mnist.pkl', 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    return train_set, valid_set, test_set

def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

def plot(tc,ec,ta,ea,size,eta,lmd,epochs,t,v):
    plt.plot(tc,color = "red",label = "training cost")
    plt.plot(ec, color = "blue", label = "evaluation_cost")
    plt.title('COST CURVE WITH SIZE ' + str(size) + " learning rate " + str(eta) + " regular term " + str(lmd) + " epochs " + str(epochs) )
    plt.legend(loc='upper left')
    plt.savefig("../figure/nn_cost_3")
    plt.show()

    plt.plot([x * 1.0 / len(t[0]) for x in ta], color = "green", label ="training_accuracy")
    plt.plot([x * 1.0/ len(v[0]) for x in ea], color = "yellow", label = "evaluation_accuracy")
    plt.title('ACCURACY CURVE WITH SIZE ' + str(size) + " learning rate " + str(eta) + " regular term " + str(lmd) + " epochs " + str(epochs) )
    plt.legend(loc='upper left')
    plt.savefig("../figure/nn_ac_3")
    plt.show()

def main(size="[784,100,10]",epochs=50,mini_batch_size=10,eta=0.1,lmd=0.001):
    size = ast.literal_eval(size)
    t,v,te = load_dataset()
    nn = NN(size)
    training_data = zip(t[0],vectorize_labels(t[1]))
    ed = zip(v[0],vectorize_labels(v[1]))
    ec,ea,tc,ta = nn.SGD(training_data, epochs, mini_batch_size, eta,
            lmbda = lmd,
            evaluation_data=ed)
    plot(tc,ec,ta,ea,size,eta,lmd,epochs,t,v)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', help='a form like [input_size, h1_size,...,output_size] default [784,100,10]')
    parser.add_argument('--epochs', help='epochs for the nerual network default 50',type=int)
    parser.add_argument('--mini_batch_size', help='mini_batch_size for nerual network default 10',type=int)
    parser.add_argument('--eta', help='eta for the nerual network default 0.1',type=float)
    parser.add_argument('--lmd', help='regularization parameter for the nerual network default 0.001',type=float)

    args = parser.parse_args()
    if not args.size or not args.epochs or not args.mini_batch_size or not args.eta or not args.lmd: 
        parser.print_help()
        sys.exit(1)
    else:
        main(size = args.size,epochs = args.epochs,mini_batch_size=args.mini_batch_size,eta=args.eta,lmd=args.lmd)
    