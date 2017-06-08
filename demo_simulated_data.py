#Import data
import oja_pca

#Generate simulated data
#Create a test set of 5 vectors which are randomly distributed. Each vector represents data from classes 1-5.
def generateData():
    np.random.seed(1)
    d = 50
    n = 40
    s1 = np.random.normal(loc = 0,scale = 5, size = (n,d))
    s2 = np.random.normal(loc = 5,scale = 5, size = (n,d))
    s3 = np.random.normal(loc = 10,scale = 1, size = (n,d))
    s4 = np.random.normal(loc = 15,scale = 1, size = (n,d))
    s5 = np.random.normal(loc = 20,scale = 1, size = (n,d))
    s =np.vstack([s1,s2,s3,s4,s5])
    s = np.concatenate((np.repeat([1,2,3,4,5],n).reshape(n*5,1),s),axis=1)
        
    print('Demo file, simulated dataset:')
    print(s)
    
    return s

##Demo functions using simulated data
if __name__ == '__main__':
    s = generateData()
    
    #Standardize data
    Z = s[:,1:51] - np.mean(s[:,1:51],axis=0) #Center the data
    p = np.size(Z,1)
    a0 = np.random.randn(p)
    a0 = a0 / np.linalg.norm(a0, axis=0)

    #Generate PCA values
    oja_pca, oja_pca_lambda = pcaGenerate(eta0=.01,X=Z,t0=1,cycle_max=50,pcCount=d)

    #Plot variance % for each principal component
    plot_PCA_Variance(oja_pca_lambda)

    #Print results
    print("1st principal component:")
    print(oja_pca[0])


# In[65]:

#Generate PCA values of simulated dataset and compare results wiht SKLearn
oja_pca, oja_pca_lambda = pcaGenerate(eta0=.01,X=Z,t0=1,cycle_max=50,pcCount=d)

#Print Principal component vectors from my oja algorithm and SKLearn
print('Compare results of PCA with SKLearn:')
pcaCompare(s[:,1:51],oja_pca[:,0:d],oja_pca_lambda[0:d])
plot_PCA_Variance(oja_pca_lambda)


# In[ ]:



