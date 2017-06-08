#Import data
import numpy as np
import sklearn.linear_model as skl
from scipy.linalg import eigh as largest_eigh
from scipy.sparse.linalg import eigsh
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from copy import deepcopy

class ojaPCA:
    #Function to compute next step in oja algorithm
    #Inputs: alpha value of current step in oja algorithm (a), current step-size (eta), feature matrix (X), current step (i)
    #Output: alpha value to intialize next step in oja algorithm with.
    def ojaStep(a,eta,X,i):
        out = a + eta*np.dot(X[i,:],np.dot(X[i,:].T,a))
        return out/np.linalg.norm(out)

    #Function to find next eta value in oja algorithm
    #Inputs: Initial value of eta (eta0), current step (t), initial step (t0)
    #Outputs: Value of next step size (eta).
    def etaNext(eta0,t,t0):
        return eta0 / (t+t0)

    #Function to run deflation on any prinicipal component
    #Inputs: Alpha vector of current step in oja algorithm (a), feature matrix (X)
    def deflation(a,X):
        return X - X.dot(np.outer(a, a))

    #Function to generate nth principal component of a set of features X
    #Inputs: Initialized vector (a), initial step-size (eta0), feature matrix (X), initial step (t0) and the max number of 
    #cycles thes the Oja algorithm should perform (cycle_max).
    #Output: Vector of length n with nth prinicipal component of X.
    #Assumptions: X is standardized.
    def ojaAlg(a0,eta0,X,t0=1,cycle_max=50):
        t = 0
        n = np.size(X,0)
        d = np.size(X,1)
        out = np.zeros((d,cycle_max+1))
        out[:,0] = a0
        a = a0
        lambdas = np.zeros(cycle_max) 
        for cycle in range(0,cycle_max):
            np.random.shuffle(X) ##Shuffles data along 1st axis (rows)
            for i in range(0,n):
                eta = etaNext(eta0,t,t0)
                a = ojaStep(a,eta,X,i)
                t += 1
            out[:,cycle+1] = a    
            lambdas[cycle] = a.dot(X.T).dot(X).dot(a)/n  #Find max eigenvalue for each eigenvector computed

        return np.mean(out[:,cycle_max-9:cycle_max],axis=1), lambdas #Returns average of last 10 PCAs, associated eigenvalues

    #Function to generate pcCount # of prinicipal components of given matrix of features (X)
    #Inputs: Initial vector of principal components (a0), initial step size (eta0), matrix of feature (x),
    #Outputs: A matrix of principal components (n x pcCount), the maximum eigenvalue of each principal component
    #Assumptions: pcCount cannot exceed the number of features in X.
    def pcaGenerate(eta0,X,t0=1,cycle_max=50,pcCount=3):
        d = np.size(X,1)    
        if pcCount > d:
            raise("pcCount must <= # of features in input matrix X")
        #Generate initial vector of a
        a0 = np.random.randn(d)
        a0 = a0 / np.linalg.norm(a0)    

        oja_pca = np.zeros(shape=(d,pcCount))
        oja_pca_lambda = np.zeros(shape=(1,pcCount))
        oja_pca[:,0], lambdas = ojaAlg(a0,eta0,deepcopy(X),t0,cycle_max) #Find PCA component
        oja_pca_lambda[0,0] = lambdas[-1] #Find top eigenvalue associated with PCA
        Z_defl  = deflation(oja_pca[:, 0], deepcopy(X))

        for i in range(1,pcCount):
            oja_pca[:,i], lambdas = ojaAlg(a0,eta0*100,deepcopy(Z_defl),t0,cycle_max)
            oja_pca_lambda[0,i] = lambdas[-1]
            Z_defl  = deflation(oja_pca[:, i], Z_defl)
        return oja_pca, oja_pca_lambda

    #Function to compare the results of my PCA generator with SKLearn PCA
    #Inputs: Feature matrix of X, results of my PCA function, a vector the maximum eigenvalue of each principal component
    #Outputs: Prints the results of my PCA generator and SKLearn
    def pcaCompare(X,pca_results,pca_lambdas):
        l = pca_results.shape[1]
        pca = PCA(l, svd_solver='randomized')
        pca.fit_transform(X)    

        for i in range(0,l):
            print('Oja alg PCA ',i,' results:',oja_pca[:,i])
            print('SKLearn PCA ',i,' results:',pca.components_[i])

    #Function to compute how many PCA vectors it takes to identify target percent of variance
    #Inputs: Results of PCA, maximum eigenvalues of each principal component, target explained variance %.
    #Outputs: Number of principal components required to reach target variance.
    #Assumptions: PCA results includes all d principal components of feature vector.
    def findPCAVariance(pca_results,pca_lambdas,target=.95):
        l = pca_lambdas.shape[0]
        tot_var = np.sum(pca_lambdas)
        comp_var = pca_lambdas[0]
        i = 1
        while (comp_var / tot_var) < target:
            comp_var += pca_lambdas[i]
            i += 1        
        return i


    #Function to plot the variance explained by each principal component
    #Input: Maximum eigenvalue of each principal component
    #Output: Plot of variance explained
    def plot_PCA_Variance(pca_lambdas):
        #Initialize values
        l = pca_lambdas.shape[0]
        tot_var = np.sum(pca_lambdas)
        comp_var = 0
        t_plot = np.empty(shape=(l, 2))

        #Fill plot data
        for i in range(0, l):
            comp_var += pca_lambdas[i]
            t_plot[i,0] = i
            t_plot[i,1] = comp_var/tot_var

        #Plot results
        fig, ax = plt.subplots()
        ax.plot(t_plot[:,1],c="green",label="variance %")
        plt.xlabel('t')
        plt.ylabel('% of total variance explained')
        plt.title('Plot of variance explained by PCA')
        plt.legend(loc='upper left')
        plt.show()


