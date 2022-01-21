# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 14:30:08 2019

Function to calculate the absorption and fluoresence spectrum 
of a single chromophore for a user defined spectral 
density (including necessary parameters)

Arguments:
(1) w=frequency (variable)

#procedural
(2) tmax: The maximum time considered in the integration

(3) w0=0-0 optical frequency
(4) osc=square of the transition dipole moment
(5) T=temperature 
(6) pigments: A string indicating the spectral density ot be used. 
(7) vargs()= tuple of parameters for the spectral density

@author: Chris


Note: only At and Aw functions are correct.
"""

import numpy as np
import lineshapes as LS
import Spectral_density as SD
from scipy.integrate import quad 
import cmath
from scipy.constants import c
from scipy.constants import hbar
from scipy.constants import h

def At(t,w0,T,pigment,vargs):
#This is the linear absorption response function
    #print("doint At")
    gt = LS.gt(t,T,pigment,vargs)[0]
    exponent = complex(-gt.real, (-1)*w0*t- gt.imag) 
    
    return(cmath.exp(exponent))



def Ft(t,w0,T,pigment,vargs):
    #print("doint Ft")
    """
    gt = LS.gt(t,T,pigment,vargs)[0]    
    lambda_D = LS.gt(t,T,pigment,vargs)[2]        #here lambda_D is the reorganization energy.   
    
    """
    gt_result = LS.gt(t,T,pigment,vargs)
    #print(gt_result)
    gt = gt_result[0]
    lambda_D = gt_result[2]
    #h_callum = 1.439
    
    exponent = complex(-gt.real, gt.imag - (w0-(2*lambda_D))*t)
    #here hbar is imported from scipy.constants
    #Should I use hbar or h??? Callum's change in function.c file
    #here in Ft function, I think I should minus the complex conjugate of gt???Yes.
    return(cmath.exp(exponent))




##
##Function to iFFT the abosorption/fluorescence spec to frequency domain.
##

def fft2w(A_time, dt):
    """
    A_time is a numpy array that contains the abosorption/fluorescence spec in time domain
    
    """
    A_freq=np.fft.ifft(A_time,norm='ortho') #a discrete fast fourier transform
    A_freq=A_freq*dt #multiply by the phase factor to account for the discretization
    #The discrete frequencies are w=n.2pi/Ndt (n=0, 1, ..., N-1)
    #The algorithm assumes dt=1 so we need to renormalize
    w = np.fft.fftfreq(A_time.size)*((2*np.pi)/dt)
    
    """
    #w = cycle per ps = ps-1   here w is in ps-1
    #to convert ps-1 to Hz:
    #1 Hz = 1 cps = 10E-12 cpps = 10E-12 ps-1
    #also:
    #1 Hz = 3.335 65 * 10E-11 wavelength_in_cm-1
    #1ps-1 = 10E12 Hz = 10E12 * 3.335 65 * 10E-11 cm-1 = 33.3565 cm-1
    
    #Finally:
    #w_in_cm-1 = w_in_ps / 33.3565 #here w in cm-1
    """
    
    #The data is oddly stored leading to plotting problems
    A_sorted=[]
    w_sorted=[]
    
    for i, item in enumerate(w):
        if item<0.0:
            w_sorted.append(item)
            A_sorted.append(A_freq[i].real)

    for i, item in enumerate(w):
        if item>=0.0:
            w_sorted.append(item)
            A_sorted.append(A_freq[i].real)
    
    return (w_sorted, A_sorted)
"""
def Ft(t,w0,T,pigment,vargs):
#this the fluorescence response function 
    
    #spectral density to be integrated to find the reorganization energy
    spec_dens=getattr(SD,pigment)
    def sd_integrand(w,vargs):
        I=spec_dens(w,vargs)/(np.pi*w)
        return I
    #Obain the reorganization energy
    L, Lerr=quad(sd_integrand,0.0,np.inf,args=(vargs))

    #conjugate line broadening line broadening function enforce mirror symmetry of fluoresnece about 0-0 
    gt=LS.gt(t,T,pigment,vargs)[0]
    gt=gt.conjugate()
    iw0t=complex(0.0,w0*t)
    
    F=cmath.exp(-iw0t-gt-(2*L))
    
    return(F, L)
"""




#Functions to transform the absorption and fluoresence to the frequency domain 
#using a Discrete Fast Fourier Transform
    
def Aw(osc,w0,T,pigment,vargs):
    #start by generating a discrete absorption response function
    dt=0.001
    timesteps=np.arange(0.0,1.0,dt) #This is a total time range of 1 ps in steps of 1 fs. This is currently hard-coded and will need fixing

    dt=dt*2.0*np.pi*c*100.0*1.0E-12 #t is now expressed not in ps but in cm as needed for the line-broadening function
    timesteps=timesteps*2.0*np.pi*c*100.0*1.0E-12 #t is now expressed not in ps but in cm as needed for the line-broadening function
    
    A_time=[] #the response function (spectrum in the time domain)
    for t in timesteps:
        A_time.append(At(t,w0,T,pigment,vargs))
    
    A_time=np.array(A_time)
        
    A_freq=np.fft.ifft(A_time,norm='ortho') #a discrete fast fourier transform
    A_freq=A_freq*dt #multiply by the phase factor to account for the discretization
    
    #The discrete frequencies are w=n.2pi/Ndt (n=0, 1, ..., N-1)
    #The algorithm assumes dt=1 so we need to renormalize
    w=np.fft.fftfreq(A_time.size)*((2*np.pi)/dt)  
    
    #The data is oddly stored leading to plotting problems
    A_sorted=[]
    w_sorted=[]
    for i, item in enumerate(w):
        if item<0.0:
            w_sorted.append(item)
            A_sorted.append(A_freq[i].real)

    for i, item in enumerate(w):
        if item>=0.0:
            w_sorted.append(item)
            A_sorted.append(A_freq[i].real)
    
    #the spectrum does not decrease to zero
    #min_corr=min(A_sorted)
    #A_sorted=A_sorted-min_corr
 
    
    
    return (w_sorted, A_sorted)


def Fw(osc,w0,T,pigment,vargs):
    #start by generating a discrete absorption response function
    dt=0.001
    timesteps=np.arange(0.0,1.0,dt) #This is a total time range of 1 ps in steps of 1 fs. This is currently hard-coded and will need fixing

    dt=dt*2.0*np.pi*c*100.0*1.0E-12 #t is now expressed not in ps but in cm as needed for the line-broadening function
    timesteps=timesteps*2.0*np.pi*c*100.0*1.0E-12 #t is now expressed not in ps but in cm as needed for the line-broadening function
    
    F_time=[] #the response function (spectrum in the time domain)
    for t in timesteps:
        F_time.append(Ft(t,w0,T,pigment,vargs))
    
    F_time=np.array(F_time)
        
    F_freq=np.fft.ifft(F_time,norm='ortho') #a discrete fast fourier transform
    F_freq=F_freq*dt #multiply by the phase factor to account for the discretization
    
    #The discrete frequencies are w=n.2pi/Ndt (n=0, 1, ..., N-1)
    #The algorithm assumes dt=1 so we need to renormalize
    w=np.fft.fftfreq(F_time.size)*((2*np.pi)/dt)  
    
    #The data is oddly stored leading to plotting problems
    F_sorted=[]
    w_sorted=[]
    for i, item in enumerate(w):
        if item<0.0:
            w_sorted.append(item)
            F_sorted.append(F_freq[i].real)

    for i, item in enumerate(w):
        if item>=0.0:
            w_sorted.append(item)
            F_sorted.append(F_freq[i].real)
    
    #the spectrum does not decrease to zero
    #min_corr=min(A_sorted)
    #A_sorted=A_sorted-min_corr
 
    
    
    return (w_sorted, A_sorted)

