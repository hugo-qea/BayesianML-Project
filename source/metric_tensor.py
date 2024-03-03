from utils import *

class MetricTensor:
    """ Metric tensor class
    """
    
    def __init__(self,sep=False) -> None:
        self.sep = sep
        
    def G(self,theta):
        pass
    
    def G_inv(self,theta):
        pass
    
class FisherInformationNormal(MetricTensor):
    """ Fisher information for normal distribution metric tensor class
    """
    
    def __init__(self,N, sep=False) -> None:
        super().__init__(sep)
        self.N = N
    def G(self,theta):
        return  (self.N/theta[1]**2) * np.diag([1,2])
    
    def G_inv(self,theta):
        return (theta[1]**2/self.N) * np.diag([1,0.5])
    

    
class BayesianInformation(MetricTensor):
    """ Metric tensor defined for the example of the Bayesian logistic regression
    """
    
    def __init__(self, alpha, X, sep=False) -> None:
        super().__init__(sep)
        self.X = X
        self.alpha = alpha
        self.N = self.X.shape[0]
        self.D = self.X.shape[1]

    def G(self, beta):
        s = self.logistic_fun(np.dot(beta.T, self.X.T))
        #diag = np.multiply(s, 1-s) # element wise multiplication
        diag = s * (1-s)
        # diag = np.dot(beta.T, self.X.T)
        # logistic_term = logistic_fun(diag)
        # diag = logistic_term * (1.0 - logistic_term)

        gamma = np.diag(diag)
        res = np.dot(self.X.T,(np.dot(gamma, self.X))) + np.eye(self.D) / self.alpha
        #print(f"determinant of G: {np.linalg.det(res)}")
        #print("sqrt G")
        #print(np.sqrt(res))
        return res
        

    def G_inv(self,beta):
        return np.linalg.pinv(self.G(beta)) + 1e-8 * np.eye(self.D)

    def logistic_fun(self, x) : 
        return 1/(1 + np.exp(-x))
    
class StochasticVM(MetricTensor):
    """ Stochastic Variational Manifold
	"""
    def __init__(self, alpha, X, sep=False) -> None:
        super().__init__(sep)
        self.X = X
        self.alpha = alpha
        self.T = self.X.shape[0]

    def G(self, beta):
        G = np.zeros((3,3))
        G[0,0] = self.T * 2 / beta[0]**2
        G[1,1] = self.T * 2 / beta[1]**2
        G[1,2] = 2* beta[2] / (beta[1] * (1 + beta[2]**2))
        G[2,1] = G[1,2]
        G[2,2] = (2 * beta[2]**2 /(1 - beta[2]**2)) + (self.T -1)/(1 - beta[2]**2)

        return G
		

    def G_inv(self,beta):
        return np.linalg.pinv(self.G(beta)) + 1e-8 * np.eye(3)

    def logistic_fun(self, x) : 
        return 1/(1 + np.exp(-x))

