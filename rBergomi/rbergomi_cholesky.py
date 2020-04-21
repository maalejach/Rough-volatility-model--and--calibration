import numpy as np
from scipy import stats
from scipy.optimize import least_squares
from scipy.stats import norm
from scipy.linalg import sqrtm
from scipy.special import hyp2f1
from scipy.interpolate import splrep, splev
import sys




class rBergomi_cholesky(object):
    """
    Class for generating paths of the rBergomi model.
    """
    def __init__(self, n = 100, N = 10000, T = 1.00, H = 0.05, eta=2.3, rho=-0.9, sigma0=0.12):
        """
        Constructor for class.
        """
        # Basic assignments
        self.T = T # Maturity
        self.n = n # Granularity (steps per year)
        self.dt = 1.0/self.n # Step size
        self.s = int(self.n * self.T) # Steps
        self.t = np.linspace(0, self.T, self.s+1)
        self.H = H # H
        self.gamma = 0.5 - H
        self.eta = eta
        self.rho = rho
        self.sigma0 = sigma0
        self.N = N # Paths
        self.cov_matrix=self.covW_Z()
        self.sqrtm_cov_matrix = sqrtm(self.cov_matrix)
    
    def covW_fun_aux(self, x):
        assert x <= 1
        return ((1 - 2 * self.gamma) / (1 - self.gamma)) * (x**(self.gamma)) * hyp2f1(1, self.gamma, 2 - self.gamma, x)

    def covW_fun(self,u, v):
        if u < v:
            return self.covW_fun(v, u)
        return v**(2*self.H) * self.covW_fun_aux(v/u)

    def covWZ_fun(self,u, v):
        H_tilde = self.H + .5
        D = np.sqrt(2*self.H) / H_tilde
        return self.rho * D * (u ** H_tilde - (u - min(u, v)) ** H_tilde)

    def covW_Z(self):
        time_range = self.t[1:]
        covWW2 = np.zeros((self.s, self.s))
        for i in range(self.s):
            for j in range(self.s):
                covWW2[i][j] = self.covW_fun(time_range[i], time_range[j])


        covWZ2 = np.zeros((self.s, self.s))
        for i in range(self.s):
            for j in range(self.s):
                covWZ2[i, j] = self.covWZ_fun(time_range[i], time_range[j])


        covZZ2 = np.zeros((self.s, self.s))
        for i in range(self.s):
            for j in range(self.s):
                covZZ2[i, j] = min(time_range[i], time_range[j])
        
        cov_matrix = np.bmat([[covWW2, covWZ2], [covWZ2.T, covZZ2]]) # matrice des covariances du vecteur (W, Z)
        return cov_matrix
    
    def simul_W_Z(self):
        G = np.random.randn(2 * self.s) # gÃ©nÃ©ration d'une gaussienne centrÃ©e rÃ©duite
        WZ_sample = np.dot(self.sqrtm_cov_matrix, G) # vecteur de mÃªme loi que (W, Z)
        W_sample, Z_sample = WZ_sample[:self.s], WZ_sample[self.s:]
        W_sample = np.insert(W_sample,0,0)
        Z_sample = np.insert(Z_sample,0,0)
        return W_sample, Z_sample
    
    def simul_S(self,S0):
        Ss = np.zeros((self.N,self.s+1))
        for i in range(self.N):
            if i % 1000 == 0:
            	print("\ri {}/{}.".format(i, self.N), end="")
            	sys.stdout.flush()
            # simulation of W and Z
            W_sample,Z_sample = self.simul_W_Z()
            # Simulation of v

            v_sample = self.sigma0**2 * np.exp(self.eta * W_sample - 0.5 * (self.eta**2) * self.t**(2*self.H))
            # Simulation of S
            int_sqrtv_dZ = np.cumsum(np.sqrt(v_sample[:-1]) * (Z_sample[1:] - Z_sample[:-1]))
            #print(int_sqrtv_dZ)
            int_sqrtv_dZ = np.insert(int_sqrtv_dZ,0,0)
            v_sample[0] = 0
            int_v_dt = np.cumsum(v_sample* self.dt)
            S = S0*np.exp(int_sqrtv_dZ - .5 * int_v_dt)
            Ss[i] = S
        return Ss
        
