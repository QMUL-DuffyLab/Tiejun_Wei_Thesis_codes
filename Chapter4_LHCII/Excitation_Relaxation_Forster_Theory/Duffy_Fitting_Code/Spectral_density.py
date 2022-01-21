# -*- coding: utf-8 -*-
"""
Created on Wed Aug 07 12:12:51 2019

Module containing functions to return the value of a spectral density C''(w)

SPEC_DENS=RENGER_CHL: The standard chlorophyll spectral density of Renger and Marcus. This based on fitting an 
empirical function to fluoresence line narrowing data. 

SPEC_DENS=CAR_2_OPT: The ansatz spectral density for carotenoids assuming  

@author: Chris#


Note: Only the spectral density functions ODO and CAR_2MODE are fully and correctly implemented. CHL_RENGER needs developing to return the reorganization energy back up the function call chain. 
"""

import numpy as np


#CHL_RENGER needs revision.
#Reorganization energy is taken from Callum. Hard coded.



def CHL_RENGER(w,vargs):
#The Chlorophyll spectral density of Renger and Marcus with default parameters.
#The frequencies do not represent real or particualr vibrational modes  
#Note: added reorganization energy for RENGER based on Callum's advise.
    S0, s1, s2=0.5, 0.8, 0.5
    w1, w2=0.56, 1.94 
    
    N=(np.pi*S0*np.power(w,5))/(s1+s2)

    c1=s1/(np.math.factorial(7.0)*2.0*np.power(w1,4))
    c1=c1*np.exp(-np.sqrt(w/w1))

    c2=s2/(np.math.factorial(7.0)*2.0*np.power(w2,4))
    c2=c2*np.exp(-np.sqrt(w/w2))

    C=N*(c1+c2)
    
    #dummy auxilliary outputs simply for generallity in the function call chain
    #Here L0 = 37.
    return(C, 37.0, 0.0)



"""
def CHL_RENGER_CUSTOM(w,vargs):
#The Chlorophyll spectral density of Renger and Marcus but now using user-defined 
#values for S0, s1, s2, w1, w2 which are given in order as the tuple vargs=()
    S0, s1, s2=vargs[0], vargs[1], vargs[2]
    w1, w2=vargs[3], vargs[4]
    
    N=(np.pi*S0*np.power(w,5))/(s1+s2)
 
    c1=s1/(np.math.factorial(7.0)*2.0*np.power(w1,4))
    c1=c1*np.exp(-np.sqrt(w/w1))

    c2=s2/(np.math.factorial(7.0)*2.0*np.power(w2,4))
    c2=c2*np.exp(-np.sqrt(w/w2))

    C=N*(c1+c2)

    #dummy auxilliary outputs simply for generallity in the function call chain
    return(C, 0.0, 0.0)

"""

def CAR_2MODE(w,vargs):
#An ansatz spectral density for a carotenoid characterized by two under-damped 
#high-frequency vibrational modes (C-C and C=C stretching) and an over-damped part representing 
#low-frequency vibrations.
    #print("Using CAR_2MODE to calculate the spectral density")
    L0, gamma0=vargs[0], vargs[1] #The reorganization energy (cm) and correlation time (fs) of the ODO
    L1, gamma1, w1=vargs[2], vargs[3], vargs[4] #Mode 1 with reorganization energy L1, correlation time (fs) gamma1 and characteristic frequency w1 (cm)
    L2, gamma2, w2=vargs[5], vargs[6], vargs[7] #Mode 2 
    
    #Typically gamma1,2,3 are given in 'fs' but actually refer to a fequency 
    #(experessed in cm^-1). The key conversion factor is that 100 fs = 53 cm^-1 

    C0=(2.0*L0*gamma0*w)/((w*w)+(gamma0*gamma0))
    C1=(2.0*L1*w*w1*w1*gamma1)/(np.power((w*w)-(w1*w1),2)+(w*w*gamma1*gamma1))
    C2=(2.0*L2*w*w2*w2*gamma2)/(np.power((w*w)-(w2*w2),2)+(w*w*gamma2*gamma2))


    #L0 is used for computing stokes shift while L1+L2 is needed to correct for 
    #shift in the spectrum due ot the over-damped modes.
    #Units: C0/C1/C2 are in same unit with w; L0/L1/L2 are in cm-1
    return(C0+C1+C2, L0, L1+L2)


def ODO(w,vargs):
#The spectral density of an over-damped Brownian oscillator with parameters
#vargs[0]=L0 is the reorganization energy in cm^-1
#vargs[1]=gamma is the correlation time of the fluctuations in cm^-1
    #print("Using ODO to calculate the spectral density")
    L0=vargs[0]
    gamma=vargs[1]
    
    C=(2.0*L0*gamma*w)/((w*w)+(gamma*gamma))

    #L0 is used for computing stokes shift. The 0.0 is just a dummy return for 
    #consistency wwith CAR_2MODE.    
    return(C, L0, 0.0)
