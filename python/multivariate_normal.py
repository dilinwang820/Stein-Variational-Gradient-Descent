import numpy as np
import numpy.matlib as nm
from svgd import SVGD

class MVN:
    def __init__(self, mu, A):
        self.mu = mu
        self.A = A
    
    def dlnprob(self, theta):
        return -1*np.matmul(theta-nm.repmat(self.mu, theta.shape[0], 1), self.A)
    
if __name__ == '__main__':
    A = np.array([[0.2260,0.1652],[0.1652,0.6779]])
    mu = np.array([-0.6871,0.8010])
    
    model = MVN(mu, A)
    
    x0 = np.random.normal(0,1, [10,2]);
    theta = SVGD().update(x0, model.dlnprob, n_iter=1000, stepsize=0.01)
    
    print "ground truth: ", mu
    print "svgd: ", np.mean(theta,axis=0)
