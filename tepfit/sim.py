# Numpy sim implementation

"""
Numpy JR model implementation
==============================
"""

"""
Importage
"""

from mpl_toolkits import mplot3d
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt




# %load jr_model.py

"""
Importages
"""


# Generic stuff

from copy import copy,deepcopy
import os,sys,glob
from itertools import product
from datetime import datetime
from numpy import exp,sin,cos,sqrt,pi, r_,floor,zeros
import numpy as np,pandas as pd
from joblib import Parallel,delayed
from numpy.random import rand,normal
import scipy


# Neuroimaging & signal processing stuff

from scipy.signal import welch,periodogram
from scipy.spatial import cKDTree


# Vizualization stuff

from matplotlib.tri import Triangulation
from matplotlib import pyplot as plt
from matplotlib import cm
import matplotlib as mpl
import seaborn as sns
import moviepy
from moviepy.editor import ImageSequenceClip
from matplotlib.pyplot import subplot



"""
Run sim function
"""

# Notes:
# - time series not returned by default (to avoid memory errors). Will be returned
#   if return_ts=True
# - freq vs. amp param sweep has default range within which max freqs are calculated
#   of 5-95 Hz


def run_net_sim(D_pyc = .0001,D_ein= .0001, D_iin = 0.0001,T = 40000,P=1, Q = 1,K = 1,Dt = 0.0001,
                dt = 0.1,gain = 20.,threshold = 0.,Pi = 3.14159,g = 0.9, noise_std=0.005,
                T_transient=10000, stim_freq=35.,stim_amp=0.5,return_ts=False,compute_connectivity=False,
                weights=None,delays=None,
                fmin = 1.,fmax = 100.,fskip = 1,
                stim_nodes=None,stim_pops=['pyc_v'],mne_welch=False,#filt_freqs = [0,np.inf],
                stim_type='sinewave',stim_on=None,stim_off=None,
                welch_nperseg=None,welch_noverlap=None,
                nu_max = 2.5,              # STANDARD JR MODEL PARAMS (JANSEN AND RITT 1995)
                v0 = 6.,
                r = 0.56,
                a_1 = 1,
                a_2 = 0.8,
                a_3 = 0.25,
                a_4 = 0.25*1,
                a = 100,
                A = 3.25,
                b = 50,
                B = 22.0,
                J = 135,
                mu = 0):

    
    """a_1 = a_1*J
    a_2 = a_2*J
    a_3 = a_3*J
    a_4 = a_4*J"""

    """
    Time units explained:


    Elements t of Ts are integers and represent 'time units'

    Physical duration of a time unit is as follows:

    Time (real, in ms) = t*dt*scaling = t*Dt

    scaling is implicitly defined as Dt/dt

    Example:  (to add)


    NOTE: need to be careful with delays units etc.

    """





    n_nodes = K

    x1 = n_nodes/2.
    x2 = x1+20.


    # Neuronal response function
    def f(u,gain=gain,threshold=threshold):
        output= 1./(1.+exp(-gain*(u-threshold)));
        return output


    def SigmoidalJansenRit(x):
      cmax = 5
      cmin = 0
      midpoint = 6
      r= 0.56
      pre = cmax / (1.0 + np.exp(r * (midpoint - x)))
      return pre

                  
    # Initialize variables
    Xi_pyc_c = np.random.uniform(low=-1.,high=1.,size=[K,T+T_transient])
    Xi_pyc_v = np.random.uniform(low=-1.,high=1.,size=[K,T+T_transient])
    Xi_ein_c = np.random.uniform(low=-1.,high=1.,size=[K,T+T_transient])
    Xi_ein_v = np.random.uniform(low=-1.,high=1.,size=[K,T+T_transient])
    Xi_iin_c = np.random.uniform(low=-1.,high=1.,size=[K,T+T_transient])
    Xi_iin_v = np.random.uniform(low=-1.,high=1.,size=[K,T+T_transient])


    pyc_c_all = np.zeros_like(Xi_pyc_c)
    pyc_v_all = np.zeros_like(Xi_pyc_v)
    ein_c_all = np.zeros_like(Xi_ein_c)
    ein_v_all = np.zeros_like(Xi_ein_v)
    iin_c_all = np.zeros_like(Xi_iin_c)
    iin_v_all = np.zeros_like(Xi_iin_v)


    Ts = np.arange(0,T+T_transient)  # time step numbers list


    if stim_type == 'sinewave':
        #state_input = ((Ts>x1+1)&(Ts<x2)).astype(float)
        state_input = 1. # not sure what this is suppose do be doing...
        stim = state_input * stim_amp*sin(2*Pi*stim_freq*Ts*Dt)
    elif stim_type == 'spiketrain':
        stim_period =1./(stim_freq*Dt)
        spiketrain = np.zeros_like(Ts)
        t_cnt = 0
        t = 0
        while t < Ts[-4]:
            t+=1
            if t>=t_cnt:
                spiketrain[t] = stim_amp
                spiketrain[t+1] = stim_amp
                spiketrain[t+2] = 0
                spiketrain[t+3] = 0
                t_cnt = t_cnt+stim_period
        stim = spiketrain
    elif stim_type== 'square':
        stim = np.zeros_like(Ts)
        stim[(Ts>=stim_on) & (Ts<=stim_off)] = stim_amp
    elif stim_type== 'rand':
        stim = np.random.randn(Ts.shape[0])*stim_amp



    # convert stim to array the size of e
    #stim = np.tile(stim,[n_nodes,1])
    
    stim_temp = np.zeros((n_nodes, stim.shape[0]))


    # arrange stimulus for the nodes
    if stim_nodes==None:
      stim = stim_temp
      pass
    else: 
        stim_temp[stim_nodes] = stim
        stim = stim_temp

        stim_nodes = np.array(stim_nodes)
        nostim_nodes = np.array([k for k in range(n_nodes) if k not in stim_nodes])
        if nostim_nodes.shape[0] != 0: stim[nostim_nodes,:] = 0.

        

    # decide which population to add the stimulus (currently mutually exclusive)
    stim_pyc_c = stim_pyc_v = stim_ein_c = stim_ein_v = stim_iin_c = stim_iin_v = np.zeros_like(stim)
    if 'pyc_v' in stim_pops: stim_pyc_v = stim
    if 'pyc_c' in stim_pops: stim_pyc_c = stim
    if 'ein_c' in stim_pops: stim_ein_c = stim
    if 'ein_v' in stim_pops: stim_ein_v = stim
    if 'iin_c' in stim_pops: stim_iin_c = stim
    if 'iin_v' in stim_pops: stim_iin_v = stim


    phi_n_scaling = (0.1 * 325 * (0.32-0.12) * 0.5 )**2 / 2.
    #+ phi_n_scaling *0.0000000001
    #Initialize just first state with random variables instead...
    """pyc_c = np.random.randn(n_nodes)
    pyc_v = np.random.randn(n_nodes)
    ein_c = np.random.randn(n_nodes)
    ein_v = np.random.randn(n_nodes) 
    iin_c = np.random.randn(n_nodes)
    iin_v = np.random.randn(n_nodes)"""

    
    pyc_c = np.zeros(n_nodes)
    pyc_v = np.zeros(n_nodes)
    ein_c = np.zeros(n_nodes)
    ein_v = np.zeros(n_nodes) + phi_n_scaling *0.00000
    iin_c = np.zeros(n_nodes)
    iin_v = np.zeros(n_nodes) 
    
    # Define time
    #ts = r_[0:sim_len:dt] # integration step time points (ms)
    #n_steps = len(ts)  #n_steps = int(round(sim_len/dt)) # total number of integration steps
    #ts_idx = arange(0,n_steps) # time step indices

    # Sampling frequency for fourier analysis
    fs = (1000.*Dt)/dt
    #fs = 1./dt


    # Make a past value index lookup table from
    # the conduction delays matrix and the
    # integration step size
    #delays_lookup = floor(delays/dt).astype(int)
    delays_lookup = floor(delays).astype(int)

    maxdelay = int(np.max(delays_lookup))+1
    
    # Integration loop
    for t in Ts[maxdelay:-1]:
        #print(t)


        # For each node, find and summate the time-delayed, coupling
        # function-modified inputs from all other nodes
        #ccin = zeros(n_nodes)
        noise_e = np.random.rand(n_nodes)
        noise_i = np.random.rand(n_nodes)
        noise_p = np.random.rand(n_nodes)

        diff = ein_c_all[:,:t] #- iin_c_all[:,:t]
        
        Ed =diff[:,::-1][np.arange(n_nodes), delays.T].T
        L = weights - 1*np.diag(weights.sum(1))
        ccen=np.sum(L*Ed.T, axis=1)
        """diff = iin_c_all[:,:t] #- iin_c_all[:,:t]
        Ed =diff[:,::-1][np.arange(n_nodes), delays.T].T
        L = weights - np.diag(weights.sum(1))
        ccin=np.sum(L*Ed.T, axis=1)
        diff = pyc_c_all[:,:t] #- iin_c_all[:,:t]
        Ed =diff[:,::-1][np.arange(n_nodes), delays.T].T
        L = weights - np.diag(weights.sum(1))
        cpin=np.sum(L*Ed.T, axis=1)"""

        """for j in range(0,n_nodes):

            # The .sum() here sums inputs for the current node here
            # over both nodes and history (i.e. over space-time)
            #insum[i] = f((1./n_nodes) * history_idx_weighted[:,i,:] * history).sum()

            for k in range(0,n_nodes):

                # Get time-delayed, sigmoid-transformed inputs
                delay_idx = delays_lookup[j,k]
                if delay_idx < t:
                    #delayed_input = f(pyc_v_all[k,t-delay_idx])
                    delayed_input = SigmoidalJansenRit(ein_c_all[k,t-delay_idx] - iin_c_all[k,t-delay_idx])
                else:
                    delayed_input = 0

                # Weight  inputs by anatomical connectivity, and normalize by number of nodes
                ccin[j]+=(1./n_nodes)*weights[j,k]*delayed_input"""


        # Cortico-cortical connections

        """gccen = g*(ccen-ccin)
        gccin = g*(ccin-ccen)/4

        gcpin = g*cpin"""
        gccen = g*(ccen)
        # Noise inputs
        """pyc_c_noise = sqrt(2.*D_pyc*dt)*Xi_pyc_c[:,t]
        pyc_v_noise = sqrt(2.*D_pyc*dt)*Xi_pyc_v[:,t]
        ein_c_noise = sqrt(2.*D_ein*dt)*Xi_ein_c[:,t]
        ein_v_noise = sqrt(2.*D_ein*dt)*Xi_ein_v[:,t]
        iin_c_noise = sqrt(2.*D_iin*dt)*Xi_iin_c[:,t]
        iin_v_noise = sqrt(2.*D_iin*dt)*Xi_iin_v[:,t]"""



               
        """

        # REFERENCE: TVB JR Model Implementation
        # https://github.com/the-virtual-brain/tvb-root/blob/017cd94f09a91b3a498efcb0213c06e79ed87ab5/scientific_library/tvb/simulator/models/jansen_rit.py#L206

            def _numpy_dfun(self, state_variables, coupling, local_coupling=0.0):
                y0, y1, y2, y3, y4, y5 = state_variables

                # NOTE: This is assumed to be \sum_j u_kj * S[y_{1_j} - y_{2_j}]
                lrc = coupling[0, :]
                short_range_coupling =  local_coupling*(y1 -  y2)

                # NOTE: for local couplings
                # 0: pyramidal cells
                # 1: excitatory interneurons
                # 2: inhibitory interneurons
                # 0 -> 1,
                # 0 -> 2,
                # 1 -> 0,
                # 2 -> 0,

                exp = numpy.exp
                sigm_y1_y2 = 2.0 * self.nu_max / (1.0 + exp(self.r * (self.v0 - (y1 - y2))))
                sigm_y0_1  = 2.0 * self.nu_max / (1.0 + exp(self.r * (self.v0 - (self.a_1 * self.J * y0))))
                sigm_y0_3  = 2.0 * self.nu_max / (1.0 + exp(self.r * (self.v0 - (self.a_3 * self.J * y0))))

                return numpy.array([
                    y3,
                    y4,
                    y5,
                    self.A * self.a * sigm_y1_y2 - 2.0 * self.a * y3 - self.a ** 2 * y0,
                    self.A * self.a * (self.mu + self.a_2 * self.J * sigm_y0_1 + lrc + short_range_coupling)
                        - 2.0 * self.a * y4 - self.a ** 2 * y1,
                    self.B * self.b * (self.a_4 * self.J * sigm_y0_3) - 2.0 * self.b * y5 - self.b ** 2 * y2,
                ])
"""        
             
        # Sigmoid transfuer functions - 'pulse to wave' 
        sigm_ein_iin = 2.0 * nu_max / ( 1. + exp(r * (v0 - (ein_c  - iin_c ))) )        
        sigm_pyc_1   = 2.0 * nu_max / ( 1. + exp(r * (v0 - (a_1 * J * pyc_c))) )
        sigm_pyc_3   = 2.0 * nu_max / ( 1. + exp(r * (v0 - (a_3 * J * pyc_c))) )
        
        pyc_u = (sigm_ein_iin  )
        ein_u = a_2*J*sigm_pyc_1+ gccen + stim_ein_c[:,t] +noise_std*noise_e #1000*np.tanh(0.0001*(g*ccen+mu)) 
        iin_u = a_4*J*sigm_pyc_3 #0*800*np.tanh(0.001*(gccin) ) +

        # Synaptic response ='wave to pulse'
        
        pyc_in = A*a*(pyc_u)                - (2.0*a*pyc_v) - ((a**2)*pyc_c)
        
        ein_in = A*a*(ein_u) - (2.0*a*ein_v) - ((a**2)*ein_c)

        iin_in = B*b*(iin_u)       - (2.0*b*iin_v) - ((b**2)*iin_c)
        
       

  
        # d/dt
        
        pyc_c   = pyc_c  + dt * (pyc_v) + 0*np.sqrt(dt)*noise_std*noise_p
        ein_c   = ein_c  + dt * (ein_v) + 0*np.sqrt(dt)*noise_std*noise_e
        iin_c   = iin_c  + dt * (iin_v) + 0*np.sqrt(dt)*noise_std*noise_i
        
        pyc_v   = pyc_v  + dt * (pyc_in)
        ein_v   = ein_v  + dt * (ein_in)
        iin_v   = iin_v  + dt * (iin_in)
        
        #print(ein_c[0:5])   
        
        pyc_v_all[:,t+1] = pyc_v
        pyc_c_all[:,t+1] = pyc_c
        ein_v_all[:,t+1] = ein_v
        ein_c_all[:,t+1] = ein_c
        iin_v_all[:,t+1] = iin_v
        iin_c_all[:,t+1] = iin_c
        

    # Concatenate sim time series
    df_all = pd.concat({'stim': pd.DataFrame(stim.T[T_transient:]),
                         'pyc_c': pd.DataFrame(pyc_c_all.T[T_transient:]),
                        'pyc_v': pd.DataFrame(pyc_v_all.T[T_transient:]),
                        'ein_c':pd.DataFrame(ein_c_all.T[T_transient:]),
                        'ein_v': pd.DataFrame(ein_v_all.T[T_transient:]),
                        'iin_c': pd.DataFrame(iin_c_all.T[T_transient:]),
                        'iin_v': pd.DataFrame(iin_v_all.T[T_transient:])}, axis=1)
    df_all.index = (Ts[T_transient:]-T_transient)*(Dt*10000.)



    return df_all
  
  


