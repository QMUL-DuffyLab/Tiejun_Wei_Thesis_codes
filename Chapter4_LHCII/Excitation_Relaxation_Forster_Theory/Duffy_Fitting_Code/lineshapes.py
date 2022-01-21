# -*- coding: utf-8 -*-
"""
Created on Wed Aug 07 13:31:15 2019

A set of functions to return the line broadening function of for a specified 
chromophore and model of the system-bath interaction. 

@author: Chris


Note: all functions here should work.
"""

import Spectral_density as SD #a list of spectral density functions
import numpy as np
from numpy import square as sq
from scipy.integrate import quad 
from scipy.integrate import nquad 
from scipy.constants import Planck as h
from scipy.constants import c
from scipy.constants import Boltzmann as kB

def gReInt(w,t,T,pigment,vargs): #integrand of the real part of the line shape function
    
    spec_dens=getattr(SD,pigment) #call spectral density function as specified by 'pigment'
    C=spec_dens(w,vargs)[0] #i.e. the actual value of C and not the auxilliary returns
    
    hbeta=(100.0*h*c)/(kB*T) #hbar/kT in cm
    
    ReInt=C*(1.0/(np.pi*sq(w)))*(1.0-np.cos(w*t))*(1/np.tanh(0.5*hbeta*w))
    
    return ReInt

def gImInt(w,t,pigment,vargs): #integrand of the imaginary part of the lineshape function

    spec_dens=getattr(SD,pigment) #call spectral density function as specified by 'pigment'
    C=spec_dens(w,vargs)[0] #The actual spec dens and not the auxilliary returns 
    
    ImInt=C*(1.0/(np.pi*sq(w)))*(np.sin(w*t)-(w*t))
    
    return ImInt
    
def gRe(t,T,pigment,vargs): #The real part of the lineshape function is explicitly dependent on temperature
    
    g1, g1err=quad(gReInt,0.0,1.0E-5,args=(t,T,pigment,vargs), limit = 100)
    g2, g2err=quad(gReInt,1.0E-5,np.inf,args=(t,T,pigment,vargs), limit = 100)
    
    return g1+g2, g1err+g2err

def gIm(t,pigment,vargs):
    
    g, gerr=quad(gImInt,0.0,np.inf,args=(t,pigment,vargs), limit = 100)
        
    return g, gerr


#gt is a general lineshape function. 
#t=time which is expressed in units of cm^-1
#'pigment' determines the appropriate spectral density fro a particular chromophore 
    #and model of the system-bath interaction
#T=temperature in K
#'vargs' is a tuple of parameters specific to a particualr spectral density 
   
#'pigment' options
# See module 'Spectral_density' for options
        
def gt(t,T,pigment,vargs):
    #print("Calling gt function...")
    #dummy call to the spctral density ot retreive the reorganization energies
    spec_dens=getattr(SD,pigment) #call spectral density function as specified by 'pigment'
    C=spec_dens(0.0,vargs)
    L0=C[1]    #L0
    aux=C[2]    #L1+L2
        
    g=complex(gRe(t,T,pigment,vargs)[0],gIm(t,pigment,vargs)[0]+(aux*t))
    gerr=complex(complex(gRe(t,T,pigment,vargs)[1],gIm(t,pigment,vargs)[1]))  

    return g, gerr, L0 #Return the reorganization energy associated with the Stokes shift


