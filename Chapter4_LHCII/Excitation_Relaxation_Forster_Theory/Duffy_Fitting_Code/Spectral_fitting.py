# -*- coding: utf-8 -*-
"""
Created on Mon Oct 07 11:53:37 2019

@author: Chris
"""

import Spectrum_mol as Spec
import numpy as np
import matplotlib.pyplot as plt

#import the spectum for Fuco in non-polar solvent
w_exp=[]
A_exp=[]
fin=open("Fuco_non_polar.txt", 'r')
for line in fin:
    exp_data=line.rstrip().split('\t')
    w_exp.append(float(exp_data[0]))
    A_exp.append(float(exp_data[1]))
fin.close()


#1150
#159
N=6.4
w0=0.0
T=300.0
pigment='CAR_2MODE'
vargs=(450.0, 53.0, 900.0, 106.0, 1536.0, 900.0, 106.0, 1150.0)
osc=1.0


'''
dt=0.001
timesteps=np.arange(0.0,1.0,dt) #This is a total time range of 1 ps in steps of 1 fs. This is currently hard-coded and will need fixing

response=[] #the response function (spectrum in the time domain)
for t in timesteps:
    response.append(At(t,w0,T,pigment,vargs))
    
plt.plot(timesteps,response)
'''

spec=Spec.Aw(osc,w0,T,pigment,vargs)[1]
waxis=Spec.Aw(osc,w0,T,pigment,vargs)[0]

#Normalization of spectra
N_exp=np.trapz(A_exp,x=w_exp)
A_exp=A_exp/N_exp

spec=N*spec

plt.plot(waxis,spec)
plt.plot(w_exp,A_exp)

fout=open('A_spec_theor.txt','w')
for i, item in enumerate(waxis):
    fout.write(str(item)+'\t'+str(spec[i])+'\n')
fout.close()

fout=open('A_spec_exp.txt','w')
for i, item in enumerate(w_exp):
    fout.write(str(item)+'\t'+str(A_exp[i])+'\n')
fout.close()

#True normalization of the theoretical spectrum for use in further calculations
N_theor=np.trapz(spec,x=waxis)
Aw_norm=spec/N_theor

fout=open('Aw_Fuco_norm.txt','w')
for i, item in enumerate(waxis):
    fout.write(str(item)+'\t'+str(Aw_norm[i])+'\n')
fout.close()
