import numpy as np 
import scipy as sp 
from scipy.special import gamma 
from scipy import stats


def f_4D(q,beta): # distribution function of a round q-Gaussian in 4D
    F = np.linspace(0,3000,100000)
    assert q>1, 'q must be greater than 1'


    term1   = -beta**2*(q-3)*(q**2-1)/4/np.pi**2
    
    if q<1.01:
        term2   = -1/(1-q)
    else:
        term2   = gamma(q/(q-1))/gamma(1/(q-1))
    
    term3 = (1 + beta*(q - 1)*F)**(1 / (1 - q) - 3/2)
    
    return term1*term2*term3, F


# Functions for random sampling in 4D 

def random_beta(F_G):
    for i in range(len(F_G)):
        beta_x = np.random.uniform(0,2*np.pi, 1)
        beta_y = np.random.uniform(0,2*np.pi, 1)
    return beta_x, beta_y



def ABEL_g_F(f_F, F):
    f_F[0] = 0
    f_F[-1] = 0
    g_F = np.pi**2*f_F*F
    return g_F

def ABEL_cdf_g(g_F, F):
    cdf_g = []
    for i in range(len(F)):
        integral = sp.integrate.trapz(y=g_F[0:i], x=F[0:i]) 
        cdf_g.append(integral)
    return cdf_g


def ABEL_F_G(Np, cdf_g, F):
    G_sample = np.random.uniform(0,0.9999999,Np) 
    y2 = sp.interpolate.interp1d(x=cdf_g, y=F, kind='nearest', )
    F_G = y2(G_sample)
    return F_G


def random_A(F_G): 
    A_x = []
    for i in range(len(F_G)):
        limit_F = np.sqrt(F_G[i])
        A_X_SQ = np.random.uniform(0,F_G[i])
        A_x.append(np.sqrt(A_X_SQ))
        A_y.append(np.sqrt(F_G[i]-A_x[i]**2))
    return A_x, A_y 


def random_beta(F_G):
    for i in range(len(F_G)):
        beta_x = np.random.uniform(0,2*np.pi, 1)
        beta_y = np.random.uniform(0,2*np.pi, 1)
    return beta_x, beta_y


# function to generate a round 4D q-Gaussian

def generate_round_4D_qgaussian(q,beta,n_part):
    
    f_F, F = f_4D(q, beta) #  4D distribution
    g_F = ABEL_g_F(f_F, F) # PDF of 4D distribution
    cdf_g = ABEL_cdf_g(g_F, F) # CDF
    F_G = ABEL_F_G(n_part, cdf_g, F) # Inverse function
    
    A_x, A_y = random_A(F_G) # random generator distributed like F_G
    
    x = []
    y = []
    px = []
    py = []

    for i in range(len(F_G)):
        beta_x = np.random.uniform(0,2*np.pi, 1)
        beta_y = np.random.uniform(0,2*np.pi, 1)
        x.append(A_x[i]*np.cos(beta_x))
        px.append(A_x[i]*(-np.sin(beta_x)))
        y.append(A_y[i]*(-np.cos(beta_y)))
        py.append(A_y[i]*(-np.sin(beta_y)))

    return x, px, y, py



