# MNIST
This is a class project which implemented different basic machine learning models by python and numpy. Each model is implemented in low level such as entory or softmax is included. For detailed report, please check

# PCA, LDA and tSNE                                          
For PCA and LDA, please go into the “project/source” directory and type                     
```bash 
python pca_lda.py 
``` 
or just open the 
``` bash
PCA and LDA.ipynb
```
file, every figure is plotted and have very detailed explaination.

For TSNE, I do not recommend to run the program, it takes several hours to finish, the figure plotted by tsne will be pasted latter. If you want to run the program, please go into the “project/source” directory and type
``` bash
python tsne.py
```

# Linear Regression                   
Go to the “project/source” directory, type
```bash
python Linear_regression.py                        
```
The solution computed by normal equation is pretty quick, can view the solution only in less a second.               

# Logistic Regression                   
Go to the “project/source” directory, type
```bash
python Logistic_regression.py --alpha 0.01 --epochs 10                         
```
You can define your own parameter set                        

# SVM                                  
Go to the “project/source” directory, type
```bash
python multiSMO.py --num_class 10 --kernel_type Gaussian_kernel --sample_data 1 
```
This program also takes a long time and much memory to run.          

# Neural Network                          
Go to the “project/source” directory, type (you can define your own parameter set )                      
```bash
python neural_network.py --size [784,100,10] --epoch 10 --mini_batch_size 10 --eta 0.01 --lmd                     
0.001 
```
