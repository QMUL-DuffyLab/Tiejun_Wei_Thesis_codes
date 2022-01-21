from pymol.cgo import *
from pymol import cmd

w = 1.25 # cylinder width 
l = 20 # cylinder length
h = 4 # cone hight
d = w * 1.618 # cone base diameter

#provide the starting xyz
s_x = 12
s_y = 16
s_z = 25


obj = [CYLINDER, s_x, s_y, s_z, s_x+l, s_y, s_z, w, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0,
       CYLINDER, s_x, s_y, s_z, s_x, s_y+l, s_z, w, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0,
       CYLINDER, s_x, s_y, s_z, s_x, s_y, s_z+l, w, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
       CONE, s_x+l, s_y, s_z, s_x+l+h, s_y, s_z, d, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 
       CONE, s_x, s_y+l, s_z, s_x, s_y+l+h, s_z, d, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 
       CONE, s_x, s_y, s_z+l, s_x, s_y, s_z+l+h, d, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]

cmd.load_cgo(obj, 'axes')