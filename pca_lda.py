# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 11:48:03 2016
PCA and LDA
@author: icecap
"""
import numpy as np
import time
import cPickle, gzip
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch

start_time = time.time()

# Load the dataset
def load_dataset():
    f = open('../data/mnist.pkl', 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    return train_set, valid_set, test_set
    
## select feature vectors with label 3, 5, 9 from MNIST data set
def select_sample():
    train_set, _, _ = load_dataset()
    three = [train_set[0][i] for i in range(len(train_set[1])) if train_set[1][i] == 3]
    five = [train_set[0][i] for i in range(len(train_set[1])) if train_set[1][i] == 5]
    nine = [train_set[0][i] for i in range(len(train_set[1])) if train_set[1][i] == 9]
    return three,five,nine

## compute means for each dimension of 28x28 = 784 dimsensions
def compute_mean(matrix):
    '''
    input: numpy matrix where each row is a sample data
    output: a numpy vector where i'th component is the mean for i'th column
    '''
    mean = np.zeros(matrix.shape[1])
    for i in range(len(mean)):
        mean[i] = np.mean(matrix.T[i])
    return mean

## compute covariance matrix
def compute_scatterMatrix(matrix,mean):
    '''
    input: `matrix' numpy matrix, each data sample is a row, `mean' numpy vector
            mean vector for the each column of the matrix
    output: a numpy matrix which is the scatter matrix of the original matrix
    '''
    sm = np.zeros((mean.size,mean.size))
    for i in range(matrix.shape[0]):
        u = (matrix[i] - mean).reshape((-1,1))
        sm += u.dot(u.T)
    return sm

## plot eigen values
def plot(s,label):
    plt.plot(s)
    plt.ylabel(label)
    plt.show()

def highest_n_eig(n,eig_pairs):
    '''
        input: n, non-negtive integer\
        eig_pairs, a list of tuple of eigenvalue and eigenvectors
        return: eigenvectors corresponding to largest n eigenvalues
    '''
    eig_vecs = np.asarray([eig_pairs[i][1] for i in range(n)]).T
    return np.asarray(eig_vecs)

def data_after_projection(x,matrix_w,mean):
    '''
    input: `x' sample data, each row is a feature vector\
    `matrix_w' principle component matrix, each row is a principle axis\
    `mean' mean vector for sample data matrix
    return: sample data after projection by matrix_w
    '''
    assert x.shape[1] == mean.size and x.shape[1] == matrix_w.shape[1]
    normalized = np.zeros_like(x)
    for i in range(x.shape[0]):
        normalized[i] = x[i] - mean
    return normalized.dot(matrix_w.T)

def data_reconstruction(mean,dataAP,matrix_w):
    '''
    input: `dataAP' data projected from original data by matrix_w,\
    each row is a projection of one sample data
    `matrix_w' projection matrix, each row is a principle axis\
    `mean' mean vector for data matrix\
    all args are numpy matrix or vectors
    return: data after reconstruction
    '''
    assert dataAP.shape[1] == matrix_w.shape[0]
    dataR = dataAP.dot(matrix_w)
    for i in range(dataR.shape[0]):
        dataR[i] += mean
    return dataR

def distortion(original_data,dataR):
    '''
    input: `original_data' numpy matrix, each row is a training sample\
    `dataR' numpy matrix, data after reconstruction
    return: the distortion between two data
    '''
    diff = original_data - dataR
    return np.sum(np.square(diff))

def different_PCA_dim_distortion(n,eig_pairs,mean,sample):
    matrix_w = highest_n_eig(n,eig_pairs).T
    dataAP = data_after_projection(sample,matrix_w,mean)
    dataR = data_reconstruction(mean,dataAP,matrix_w)
    distor = distortion(sample,dataR)
    return distor

def plot_sample_data(sample,mean,three,five,nine,eig_pairs):
    matrix_w = highest_n_eig(2,eig_pairs).T
    dataAP = data_after_projection(sample,matrix_w,mean).T
    class_1_size = len(three)
    class_2_size = len(five)
    class_3_size = len(nine)
    range_1 = class_1_size
    range_2 = class_1_size + class_2_size
    range_3 = class_1_size + class_2_size + class_3_size
    
    assert dataAP.shape[1] == class_1_size + class_2_size + class_3_size
    plt.plot(dataAP[0,0:range_1], dataAP[1,0:range_1], 'o',\
             markersize=7, color='blue', alpha=0.5, label='class1')
    plt.plot(dataAP[0,range_1:range_2], \
             dataAP[1,range_1:range_2], '*',\
             markersize=7, color='red', alpha=0.5, label='class2')
    plt.plot(dataAP[0,range_2:range_3], \
             dataAP[1,range_2:range_3], 'x',
             markersize=7, color='yellow', alpha=0.5, label='class3')
    plt.xlabel('x_values')
    plt.ylabel('y_values')
    plt.legend()
    plt.title('Transformed samples with class labels')

def compute_within_class_scatter_matrix(class_list):
    '''
    input: a list of data sample
    output: within_class_scatter_matrix wcsm
    '''
    wcsm_size = len(class_list[0][0])
    wcsm = np.zeros((wcsm_size,wcsm_size))
    for i in range(len(class_list)):
        tem = np.zeros((wcsm_size,wcsm_size))
        mean = compute_mean(np.asarray(class_list[i]))
        tem = compute_scatterMatrix(np.asarray(class_list[i]),mean)
        wcsm += tem
    return wcsm

def compute_between_class_scatter_matrix(class_list):
    '''
    input: a list of data sample
    output: between_class_scatter_matrix wcsm
    '''
    over_all_matrix = []
    for matrix in class_list:
        over_all_matrix += matrix
    overall_mean = compute_mean(np.asarray(over_all_matrix)).reshape((-1,1))
    bcsm_size = len(overall_mean)
    bcsm = np.zeros((bcsm_size,bcsm_size))
    for matrix in class_list:
        mean = compute_mean(np.asarray(matrix)).reshape((-1,1))
        sample_size = np.asarray(matrix).shape[0]
        bcsm += sample_size * (mean - overall_mean).dot((mean-overall_mean).T)
    return bcsm

def main():
    print "COMPUTING FOR PCA"
    three,five,nine = select_sample()
    sample = np.asarray(three+five+nine)
    print "Laoding data set done"
    mean = compute_mean(sample)
    print "Compute sample mean done"
    scatterMatrix = compute_scatterMatrix(sample,mean)
    print "Compute scatterMatrix done"
    eig_val_sc, eig_vec_sc = np.linalg.eig(scatterMatrix)
    print "Compute eigenvalues and eigenvectors for PCA done"
    eig_pairs = [(np.abs(eig_val_sc[i]), eig_vec_sc[:,i]) for i in range(len(eig_val_sc))]
    eig_pairs.sort(key=lambda x: x[0], reverse=True)
    print "Sort eig_pairs by eigenvalues done"
    vals = [i[0] for i in eig_pairs]
    print "plotting eigenvalues for PCA"
    time.sleep(5)
    plot(vals," eigenvalues for PCA")
    plt.savefig('../figure/ePCA.pdf')
    distortion_list = [2,10,50,100,200,300]
    distor = [different_PCA_dim_distortion(i,eig_pairs,mean,sample) for i in distortion_list]
    print "compute distortion after projection and reconstruction done"
    print "plotting distortion"
    time.sleep(5)
    plot(distor,"distortion after pca projection and reconstruction")
    plt.savefig('../figure/dPCA.png')
    print "plotting data after pca projection and reconstruction"
    time.sleep(5)
    plot_sample_data(sample,mean,three,five,nine,eig_pairs)
    plot([],"")
    plt.savefig('../figure/daPCA.pdf')

    ########################################### LDA #################################

    print
    print "COMPUTING FOR LDA"
    class_list = [three,five,nine]
    wcsm = compute_within_class_scatter_matrix(class_list)
    print "compute within_class_scatter_matrix done"
    bcsm = compute_between_class_scatter_matrix(class_list)
    print "compute between_class_scatter_matrix done"

    eig_vals, eig_vecs = np.linalg.eig(np.linalg.pinv(wcsm).dot(bcsm))
    print "compute eigenvalues and eigenvectors for LDA done"
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
    eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)
    print "Sort eig_pairs by eigenvalues done"
    vals = [i[0] for i in eig_pairs]

    print "ploting eigenvalues for LDA"
    time.sleep(5)
    plot(vals[:100],"eigenvalues for LDA")
    plt.savefig('../figure/eLDA.pdf')
    print "ploting LDA data"
    time.sleep(5)
    plot_sample_data(sample,compute_mean(np.asarray(three+five+nine)),three,five,nine,eig_pairs)
    plot([],"")
    plt.savefig('../figure/daLDA.pdf')
    print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == "__main__":
    main()