"""

This script takes in coordinates( using the load_coor function in the load_mol2_data.py)
and the list of corresponding TRESP charges (which is generated mannually by fixing the index)
to calculate the coupling between two molecules

this version is calculating lut620 and cla612


The Hamiltonian construction:

H =     [ E620  J       J       J   ]
        [       E610    J       J   ]
        [               E611    J   ]
        [                       E612]


14900 Hamiltonian_result.npy
15300 Hamiltonian_result2.npy
15100 Hamiltonian_result3.npy
14700 4
14500 5
14800

chl 15100, lut 14500 Hamiltonian_result8.npy
"""
from natsort import natsorted, ns
import pickle
import csv
import numpy as np
from load_mol2_data import *
import os
from math import *
import matplotlib.pyplot as plt


E_lut620 = 14500
E_cla610 = 15100
E_cla611 = 15100
E_cla612 = 15100

E_0 = 8.85E-12   #perimittivity of free space unit:F/m
E_r = 2      #the dielectric constant in protein. unitless.
elemental_charge = 1.6E-19

def calc_distance(list1, list2):
    x1,y1,z1 = list1
    x2,y2,z2 = list2
    return (sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2))

def point2point(np_array1, np_array2):
    coor1 = np_array1[0:3]
    coor2 = np_array2[0:3]
    tresp1 = np_array1[3:4]
    tresp2 = np_array2[3:4]
    p2p = (tresp1*tresp2)/abs(calc_distance(coor1,coor2))
    return p2p

def coupling_calc(mol1, mol2):
    summation = 0
    for atom1 in mol1:
        for atom2 in mol2:
            summation += point2point(atom1, atom2)
    #print(summation)
    constant = (elemental_charge*elemental_charge) / (E_0 * E_r * 4 * pi * 1.0E-10)
    coupling = summation * constant * 5.034E22    #Unit is Joules. Note that 1 Joules = 5.034 45 x 10+22 cm-1.
    return coupling


if __name__ == '__main__':
    Hamiltonian_result = []

    lut620_dir = "LUT620_FRAMES/"
    cla612_dir = "CLA612_FRAMES/"
    cla611_dir = "CLA611_FRAMES/"
    cla610_dir = "CLA610_FRAMES/"

    lut620_dirlist = os.listdir(lut620_dir)
    cla612_dirlist = os.listdir(cla612_dir)
    cla611_dirlist = os.listdir(cla611_dir)
    cla610_dirlist = os.listdir(cla610_dir)

    """
    sort the dirlist, so that it appears in 1,2,3,4,5...
    instead of 1,10,100,101, etc.
    """

    lut620_dirlist = natsorted(lut620_dirlist, key=lambda y: y.lower())
    cla612_dirlist = natsorted(cla612_dirlist, key=lambda y: y.lower())
    cla611_dirlist = natsorted(cla611_dirlist, key=lambda y: y.lower())
    cla610_dirlist = natsorted(cla610_dirlist, key=lambda y: y.lower())
    print(lut620_dirlist, cla610_dirlist, cla611_dirlist, cla612_dirlist)
    #all = [lut620_dirlist,cla612_dirlist,cla611_dirlist,cla610_dirlist]


    for job_id in range(0,1000):
        lut620_file_dir = lut620_dir + lut620_dirlist[job_id]
        cla612_file_dir = cla612_dir + cla612_dirlist[job_id]
        cla611_file_dir = cla611_dir + cla611_dirlist[job_id]
        cla610_file_dir = cla610_dir + cla610_dirlist[job_id]

        #print(lut_file_dir,cla_file_dir)
        lut620_atoms = np.asarray(load_coor(lut620_file_dir))  # lutein atom coordinates in [98,3] np array
        cla612_atoms = np.asarray(load_coor(cla612_file_dir))  # cla in [137,3] np array
        cla611_atoms = np.asarray(load_coor(cla611_file_dir))
        cla610_atoms = np.asarray(load_coor(cla610_file_dir))
        # print(lut_atoms)
        # print(lut_atoms.shape, charges.shape)
        # print(len(np.loadtxt("lut_md_tresp_list.txt")))
        lut620 = np.hstack((lut620_atoms, np.loadtxt("lut_md_tresp_list.txt").reshape(len(np.loadtxt("lut_md_tresp_list.txt")), 1)))
        cla612 = np.hstack((cla612_atoms, np.loadtxt("cla_md_tresp_list.txt").reshape(len(np.loadtxt("cla_md_tresp_list.txt")), 1)))
        cla611 = np.hstack((cla611_atoms, np.loadtxt("cla_md_tresp_list.txt").reshape(len(np.loadtxt("cla_md_tresp_list.txt")), 1)))
        cla610 = np.hstack((cla610_atoms, np.loadtxt("cla610_md_tresp_list.txt").reshape(len(np.loadtxt("cla610_md_tresp_list.txt")), 1)))

        print(lut620.shape, cla612.shape, cla611.shape, cla610.shape)
        #print(job_id+1, coupling_calc(lut,cla))

        Hamiltonian = np.zeros((4, 4))
        #print(Hamiltonian)

        Hamiltonian[0,0] = E_lut620
        Hamiltonian[1,1] = E_cla610
        Hamiltonian[2,2] = E_cla611
        Hamiltonian[3,3] = E_cla612

        print("calculating coupling and constructing Hamiltonian for frame %s" %job_id)
        Hamiltonian[0, 1] = coupling_calc(lut620, cla610)
        Hamiltonian[0, 2] = coupling_calc(lut620, cla611)
        Hamiltonian[0, 3] = coupling_calc(lut620, cla612)
        Hamiltonian[1, 2] = coupling_calc(cla610, cla611)
        Hamiltonian[1, 3] = coupling_calc(cla610, cla612)
        Hamiltonian[2, 3] = coupling_calc(cla611, cla612)

        Hamiltonian_result.append(Hamiltonian)
    #result_norm = [(float(i)-min(result))/(max(result)-min(result)) for i in result]      #normalizing the list to let elements to be [0,1]
    np.save('Hamiltonian_result8.npy', Hamiltonian_result)
    print("saving result to 'Hamiltonian_result8.npy'")
"""
    with open('coupling_result.csv', 'w') as f:
        wr = csv.writer(f, quoting=csv.QUOTE_ALL)
        wr.writerow(result)

    with open("coupling_result_normed.csv", "w") as f:
        wr = csv.writer(f, quoting=csv.QUOTE_ALL)
        wr.writerow(result_norm)
"""
"""
    plt.plot(np.arange(0,1000), Hamiltonian_result[:,1,1], 'ro')
    plt.axis([0, 1000, 0, 1.2])
    plt.show()

    plt.plot(np.arange(0, 1000), result, 'ro')
    plt.axis([0, 1000, 0, 1.2])
    plt.show()
"""



