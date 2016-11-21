from smo import *
import threading
import argparse
import sys


def correct_ratio(hx,y):
    r = sum(np.equal(hx,y))*1.0 / len(y)
    return r

def sortByLabel(numOfC):

    train, valid, test = load_dataset()
    tx,ty = train[0], train[1]
    tex,tey = test[0],test[1]
    f_x,f_tx = [],[]
    for i in range(numOfC):
        f_x.append(FilterByLabel(tx,ty,i))
        f_tx.append(FilterByLabel(tex, tey, i))
    return f_x, f_tx

def FilterByLabel(data, labelMat, labels):
    fLabelMat = [label if label == labels else -1 for label in labelMat]
    fdata  = [data[n] for n in range(len(data)) if fLabelMat[n]!=-1]
    fLabelMat = [n for n in fLabelMat if n >0]
    return fdata

def labelToIndex(pre,i,k):
    index = []
    for e in pre:
        if e > 0: index += [i]
        else: index += [k]
    return index

class MultiClassSMO():

    def __init__(self, numOfC,kernel_type):
        self.SMOS = [SMO(kernel_type) for i in range((numOfC)*(numOfC))]
        self.classifierMatrix = np.array(self.SMOS).reshape((numOfC),(numOfC))
        self.numOfC = numOfC
    

    def train_SMOS(self,data):

        threads = []
        for i in range(self.numOfC):
            for k in range(self.numOfC):
                if(i < k):
                    X_train,Y_train = randomizeX(data[i],data[k])
                    t = threading.Thread(target= self.classifierMatrix[i][k].train, 
                                            args=(X_train,Y_train))
                    t.start()
                    threads.append(t)

        for t in threads:
            t.join()

    def predit_all_smos(self,test):
        test_data = []
        for i in range(0,self.numOfC):
            test_data += test[i]

        test_label = []
        for i in range(0,self.numOfC):
            test_label += [i]*len(test[i])

        Voter_matrix = []
        for i in range(self.numOfC):
            for k in range(self.numOfC):
                if(i < k):
                   svm_ik = self.classifierMatrix[i][k]
                   result = svm_ik.predict(np.array(test_data))
                   Voter_matrix.append(labelToIndex(result,i,k))

        self.Voter_matrix = Voter_matrix
        score = self.compute_score()
        print correct_ratio(score, test_label)

    def compute_score(self):
        v_m = np.matrix(self.Voter_matrix).T
        score = [np.argmax(np.bincount(np.squeeze(np.asarray(v_m[i])))) for i in range(v_m.shape[0])]
        return score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_class', help='the number of classes to be classified',type=int)
    parser.add_argument('--kernel_type',help='the type of kernel_type to use, either linear_kernel or Gaussian_kernel')
    parser.add_argument('--sample_data',help='whether sample data or not, 1 yes, 0 no',type=int)
    args = parser.parse_args()
    if not args.num_class or not args.kernel_type: 
        parser.print_help()
        sys.exit(1) 
    else:
        sample_data = []
        data, test = sortByLabel(10)
        if args.sample_data:
            for i in range(10): sample_data.append(data[i][:1000])
            data = sample_data

        classifer = MultiClassSMO(numOfC=args.num_class,kernel_type=args.kernel_type)
        classifer.train_SMOS(data)
        classifer.predit_all_smos(test)
