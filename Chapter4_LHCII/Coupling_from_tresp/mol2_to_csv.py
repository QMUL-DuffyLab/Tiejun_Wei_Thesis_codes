"""
This script is to convert mol2 files generated from AMBER CPPTRAJ to csv files.
Together with its tresp charges, see below:

X   Y   Z   TrESP

N*4 array like data

"""

from natsort import natsorted, ns
import numpy as np
from load_mol2_data import *
import os
import csv


#Look for all directories with keyword"FRAME" in its name and sort it in natural order


if __name__ == '__main__':
    dirlist = ['CLA610_FRAMES/']
    for dir in dirlist:
        #print(dir)
        files_under_dir = os.listdir(dir)
        files_under_dir = natsorted(files_under_dir, key=lambda y: y.lower())
        #print(files_under_dir)
        for job_id in range(0,1000):
            files_path = dir + files_under_dir[job_id]
            #print(files_path)
            atoms = np.asarray(load_coor(files_path))
            atoms_with_tresp = np.hstack((atoms, np.loadtxt("cla610_md_tresp_list.txt").reshape(len(np.loadtxt("cla610_md_tresp_list.txt")), 1)))
            print(job_id+1, atoms_with_tresp.shape)
            np.save("")
            with open("CLA610_CSV/"+"frame{}.txt".format(job_id+1), 'w') as f:
                np.savetxt(f, atoms_with_tresp, fmt=' % 016.8e')
                #wr = csv.writer(f, quoting=csv.QUOTE_ALL)
                #wr.writerow(atoms_with_tresp)




