# MNIST
PART 1 INSTRUCTIONS FOR PROGRAM                              
PCA, LDA and TSNE                                          
For PCA and LDA, please go into the “project/source” directory and ​ type                     
python ​ pca_lda.py                     
The program takes about 87 seconds to finish and output what the program is doing step                         
by step, the related figure will be plotted one by one by matplot.                             
For TSNE, I do not recommend to run the program, it takes several hours to finish, the                  
figure plotted by tsne will be pasted latter. If you want to run the program, please go into the                        
“project/source” directory and ​ type                      
python tsne.py                         
LINEAR REGRESSION                   
Go to the “project/source” directory, t ​ ype                        
python Linear_regression.py                        
The solution computed by normal equation is pretty quick, can view the solution only in                        
less a second.               
LOGISTIC REGRESSION                   
Go to the “project/source” directory, t ​ ype                            
python Logistic_regression.py --alpha 0.01 --epochs 10                         
You can define your own parameter set                        
SVM                                  
Go to the “project/source” directory, t ​ ype                      
python multiSMO.py --num_class 10 --kernel_type Gaussian_kernel                          
--sample_data 1 ​ This program also takes a long time and much memory to run.                        
Neural Network                          
Go to the “project/source” directory, t ​ ype (​ you can define your own parameter set ) ​                      
python neural_network.py --size [784,100,10] --epoch 10 --mini_batch_size 10 --eta 0.01 --lmd                     
0.001 
