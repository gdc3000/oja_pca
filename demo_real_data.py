#Import data
import oja_pca

#Function to import real data from hitters dataset
def import_data():
    #Download hitters data
    hitters = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Hitters.csv', sep=',', header=0).dropna()
    
    #Format data
    hitters = hitters.dropna()
    Y = hitters.Salary
    X = hitters.drop(['Salary'], 1)
    X = pd.get_dummies(X,drop_first=True)
    
    #Normalize X and center Y
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    Y = np.array(Y - np.mean(Y))
    
    return X, Y

#Demo functions using real data
if __name__ == '__main__':
    np.random.seed(1)
    X, Y = import_data()
    n, d = X.shape
    a0 = np.random.randn(d)
    a0 = a0 / np.linalg.norm(a0, axis=0)

    #Generate PCA values
    oja_pca, oja_pca_lambda = pcaGenerate(eta0=.1,X=X,t0=1,cycle_max=50,pcCount=d)

    #Plot variance % for each principal component
    plot_PCA_Variance(oja_pca_lambda)

    #Print results
    print("1st principal component:")
    print(oja_pca[0])
