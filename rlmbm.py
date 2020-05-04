#simulation of Riemann-Liouville Multi-fractional Brownian Motion (RLMBM) from 
#'Modelling of locally self-similar processes using multifractional Brownian motion of Riemann-Liouville type' by S. V. Muniandy and S. C. Lim (2001), Phys. Rev. E 63, 046104
#Daniel Han 3.5.2020

import numpy as np
from scipy.special import gamma
import matplotlib.pyplot as plt

#weighting function
def weighting(t,dt,H):
    w = (1/gamma(H+0.5))*((pow(t,H+0.5)-pow(t-dt,H+0.5))/((H+0.5)*dt))
    return w

#improved weighting function from S. Rambaldi and O. Pinazza (1994) Physica D 208, 21
def weighting_improved(t,dt,H):
    w = (1/gamma(H+0.5))*np.sqrt((pow(t,2*H)-pow(t-dt,2*H))/(2*H*dt))
    return w

#set parameters
mu = 0.
sigma = 1.
N = 1000
dt = 0.1
#do square root of dt to save time in calculation
sqrtdt = np.sqrt(dt)
#make time array
t = np.array([i*dt for i in range(1,N+1)])
#Hurst exponent array with particular hurst exponent function
H = 0.5+0.5*np.sin((1/2)*t)
# H = 0.9*np.ones_like(t)
# for i in range(0,200):
#     H[i] = 0.1
# for i in range(800,N):
#     H[i] = 0.4
#Brownian increments array
xi = np.random.normal(mu,sigma,N)
#weights array
weights = [weighting_improved(t[i],dt,H[i]) for i in range(0,N)]
#container for MBM values
BH = np.zeros_like(t)
#container for BM values
B = np.zeros_like(t)
#sum Brownian exponents and weights to make MBM
BH[0] = xi[0]*weights[0]*np.sqrt(dt)
for j in range(1,N):
    wtemp = np.flip(weights[:j])
    xtemp = xi[:j]
    BH[j] = np.sum(np.multiply(wtemp,xtemp))*sqrtdt
    B[j] = np.sum(xtemp)*sqrtdt

#plot figure
plt.figure()
plt.subplot(211)
plt.plot(t,B,alpha=0.5,label=r'$B(t)$')
plt.plot(t,BH,alpha=0.5,label=r'$B_H(t)$')
plt.legend()
plt.ylabel('x')
plt.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False) 
plt.subplot(212)
plt.plot(t,H)
plt.ylabel('H')
plt.xlabel('t')
plt.tight_layout()
plt.show()