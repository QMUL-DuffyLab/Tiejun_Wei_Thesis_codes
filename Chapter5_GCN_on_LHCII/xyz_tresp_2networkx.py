#The main frame are taken from github repo: https://github.com/zotko/xyz2graph
#Modified by TW to adapt in-house generated data with tresp charges and convert to networkx style graph
#With some basic visualization functions.

import re
from itertools import combinations
from math import sqrt
import math
import pandas as pd
import networkx as nx
import numpy as np



atomic_radii = dict(Ac=1.88, Ag=1.59, Al=1.35, Am=1.51, As=1.21, Au=1.50, B=0.83, Ba=1.34, Be=0.35, Bi=1.54, Br=1.21,
                    C=0.68, Ca=0.99, Cd=1.69, Ce=1.83, Cl=0.99, Co=1.33, Cr=1.35, Cs=1.67, Cu=1.52, D=0.23, Dy=1.75,
                    Er=1.73, Eu=1.99, F=0.64, Fe=1.34, Ga=1.22, Gd=1.79, Ge=1.17, H=0.23, Hf=1.57, Hg=1.70, Ho=1.74,
                    I=1.40, In=1.63, Ir=1.32, K=1.33, La=1.87, Li=0.68, Lu=1.72, Mg=1.10, Mn=1.35, Mo=1.47, N=0.68,
                    Na=0.97, Nb=1.48, Nd=1.81, Ni=1.50, Np=1.55, O=0.68, Os=1.37, P=1.05, Pa=1.61, Pb=1.54, Pd=1.50,
                    Pm=1.80, Po=1.68, Pr=1.82, Pt=1.50, Pu=1.53, Ra=1.90, Rb=1.47, Re=1.35, Rh=1.45, Ru=1.40, S=1.02,
                    Sb=1.46, Sc=1.44, Se=1.22, Si=1.20, Sm=1.80, Sn=1.46, Sr=1.12, Ta=1.43, Tb=1.76, Tc=1.35, Te=1.47,
                    Th=1.79, Ti=1.47, Tl=1.55, Tm=1.72, U=1.58, V=1.33, W=1.37, Y=1.78, Yb=1.94, Zn=1.45, Zr=1.56)



#Here starts the Graph Object
class MolGraph:
    """Represents a molecular graph."""
    __slots__ = ['elements', 'x', 'y', 'z', 'tresp', 'molname', 'adj_list',
                 'atomic_radii', 'bond_lengths']

    def __init__(self):
        self.elements = []
        self.x = []
        self.y = []
        self.z = []
        self.adj_list = {}
        self.atomic_radii = []
        self.bond_lengths = {}
        self.tresp = []  #added tresp charges, same format as xyz coor.
        self.molname = [] # added molname for coloring in graph

        
    def read_1callumcsv(self, file_path):
        """
        take in a callum's csv file. forexample:
        
            CLA610
            CLA611
            CLA612
            LUT620
            
        then take the corresponding raw data files from each dir and generate a molgraph.
    
        data in the raw data files:
        atom name (need to be truncated); x;y;z; tresp.
        
        """
        data = pd.read_csv(file_path, header=None, delim_whitespace=True, keep_default_na=False, na_values=[''])
        #print(data)
        
        with open(file_path) as file:
            for element, x, y, z, tresp in data.values:

                element = element[0]
                if element == 'M':
                    element = 'Mg'
                    
                #print(element)
                #print(x,y,z)
                #print(tresp)
                self.elements.append(element)
                self.x.append(float(x))
                self.y.append(float(y))
                self.z.append(float(z))
                self.tresp.append(float(tresp))
        self.atomic_radii = [atomic_radii[element] for element in self.elements]
        self._generate_adjacency_list()
        
    def read_4callumcsv(self, file_path):
        """
        take in a list callum's csv file. forexample:
        
        file_path = ["CLA610/file1", "CLA611/file1"..."LUT620/file1"]
            
        then take the corresponding raw data files from each dir and generate a big molgraph all together.
    
        data in the raw data files:
        atom name (need to be truncated); x;y;z; tresp.
        
        
        USAGE:
        mg4 = MolGraph()
        test_file_list = ['./data/CLA610/frame1.csv', './data/CLA611/frame1.csv', './data/CLA612/frame1.csv', './data/LUT620/frame1.csv']
        mg4.read_4callumcsv(test_file_list)
        
        """
        data_1 = pd.read_csv(file_path[0], header=None, delim_whitespace=True, keep_default_na=False, na_values=[''], names = ['a', 'b', 'c', 'd', 'e', "f"])
        data_2 = pd.read_csv(file_path[1], header=None, delim_whitespace=True, keep_default_na=False, na_values=[''], names = ['a', 'b', 'c', 'd', 'e', "f"])
        data_3 = pd.read_csv(file_path[2], header=None, delim_whitespace=True, keep_default_na=False, na_values=[''], names = ['a', 'b', 'c', 'd', 'e', "f"])
        data_4 = pd.read_csv(file_path[3], header=None, delim_whitespace=True, keep_default_na=False, na_values=[''], names = ['a', 'b', 'c', 'd', 'e', "f"])

        
        #print("shapes are here")
        #print(data_1.shape[0])
        #print(data_4.shape[0])
        
        ###
        #Add a section here to seperate the element, xyz, tresp data and the CONNECTION data.
        ###
        
        data_1_structure, data_1_CONNECT = split_structure_connect(data_1)
        data_2_structure, data_2_CONNECT = split_structure_connect(data_2)
        data_3_structure, data_3_CONNECT = split_structure_connect(data_3)
        data_4_structure, data_4_CONNECT = split_structure_connect(data_4)
        
        
        #here we remove the excess column of data_structure
        #print(type(data_1_structure))
        #print(data_1_structure)
        data_1_structure = data_1_structure.drop(labels='f', axis=1)
        data_2_structure = data_2_structure.drop(labels='f', axis=1)
        data_3_structure = data_3_structure.drop(labels='f', axis=1)
        data_4_structure = data_4_structure.drop(labels='f', axis=1)
        
        #further check the dataframe passed.
        #print("The shape of the structure data loaded for this frame are:")
        #for element in (data_1_structure, data_2_structure, data_3_structure, data_4_structure):
            #print(element.shape)

        
        data_all = pd.concat([data_1_structure, data_2_structure, data_3_structure, data_4_structure])
        list_of_CONNECT_data = (data_1_CONNECT, data_2_CONNECT, data_3_CONNECT, data_4_CONNECT)
        
        #print(data_all)
        
        num = 0
        label = ""
        
        for element, x, y, z, tresp in data_all.values:
            element = element[0]
            if element == 'M':
                element = 'Mg'    
            #print(element)
            #print(x,y,z)
            #print(tresp)
            self.elements.append(element)
            self.x.append(float(x))
            self.y.append(float(y))
            self.z.append(float(z))
            self.tresp.append(float(tresp))
            if num < data_1.shape[0]:
                label = 'CLA'
            elif num < data_1.shape[0] + data_2.shape[0]:
                label = 'CLA'
            elif num < data_1.shape[0] + data_2.shape[0] + data_3.shape[0]:
                label = 'CLA'
            else:
                label = 'LUT'
            self.molname.append(label)
            num += 1
        self.atomic_radii = [atomic_radii[element] for element in self.elements]
        self._generate_adjacency_list_bond(list_of_CONNECT_data)
    
    
    def read_xyz(self, file_path: str) -> None:
        """Reads an XYZ file, searches for elements and their cartesian coordinates
        and adds them to corresponding arrays."""
        pattern = re.compile(r'([A-Za-z]{1,3})\s*(-?\d+(?:\.\d+)?)\s*(-?\d+(?:\.\d+)?)\s*(-?\d+(?:\.\d+)?)')
        with open(file_path) as file:
            for element, x, y, z in pattern.findall(file.read()):
                self.elements.append(element)
                self.x.append(float(x))
                self.y.append(float(y))
                self.z.append(float(z))
        self.atomic_radii = [atomic_radii[element] for element in self.elements]
        self._generate_adjacency_list(list_of_CONNECT_data)

    def _generate_adjacency_list(self):
        """Generates an adjacency list from atomic cartesian coordinates."""
        node_ids = range(len(self.elements))
        for i, j in combinations(node_ids, 2):
            x_i, y_i, z_i = self.__getitem__(i)[1]
            x_j, y_j, z_j = self.__getitem__(j)[1]
            distance = sqrt((x_i - x_j) ** 2 + (y_i - y_j) ** 2 + (z_i - z_j) ** 2)
            if 0.1 < distance < (self.atomic_radii[i] + self.atomic_radii[j]) * 1.3:
                self.adj_list.setdefault(i, set()).add(j)
                self.adj_list.setdefault(j, set()).add(i)
                self.bond_lengths[frozenset([i, j])] = round(distance, 5)
                

    def _generate_adjacency_list_bond(self, list_of_CONNECT_data):
        """
        generate adj matrix from the bond list, the same file callum provides.
        Takes in the list of data_CONNECT
        
        generate the adjacency matrix based on each element,
        then summarize to get the sum_adj_matrix using function "sum_adj_matrix()"
        
    
        """
    
        def generate_adj_bond(CONNECT_data):
            edge_length = CONNECT_data.shape[0]
        
            #print(edge_length)
            adj_mat = np.zeros([edge_length, edge_length])
        
            for line in CONNECT_data.iterrows():
                #print(line[1]["b"]) #this is the starting atom, e.g. from 1 to 136 
                i=0 #initialize the idx number for the next for loop.
                for element in line[1][1:]: # skip the CONNECT element, the rest element in this for loop are need to be connected.
                    if str(element) != "nan": # skip the nan elements.
                        if i == 0: #i.e. the start of the for loop. the first element.
                            atom_start = int(element)
                            #print("the starting atom is:")
                            #print(atom_start)
                            i+=1
                        else: #the second, third... fourth of the element.
                            atom_connect = int(element)
                            #print("the connecting atom is:")
                            #print(atom_connect)
                    
                            adj_mat[atom_start-1,atom_connect-1] = 1
                            adj_mat[atom_connect-1, atom_start-1] = 1

                            i+=1
            return adj_mat
    
        list_of_small_adj_mat = []
    
        for CONNECT_data in list_of_CONNECT_data:
            #print(CONNECT_data.shape)
            list_of_small_adj_mat.append(generate_adj_bond(CONNECT_data))
        
        big_adj_mat = sum_adj_matrix(list_of_small_adj_mat)
        
        #generating bond length here.
        for col in range(len(big_adj_mat)):
            for row in range(len(big_adj_mat)):
                if big_adj_mat[col][row] == 1:
                    #print("operating on: %d, %d" %(col, row) )
                    x_i, y_i, z_i = self.__getitem__(col)[1]
                    x_j, y_j, z_j = self.__getitem__(row)[1]
                    distance = sqrt((x_i - x_j) ** 2 + (y_i - y_j) ** 2 + (z_i - z_j) ** 2)
                    #print(distance)
                    self.bond_lengths[frozenset([col, row])] = round(distance, 5) 
                    #add the calculated bondlength to the bond length matrix: self.bond_lengths
    
        #return (big_adj_mat, list_of_small_adj_mat) #arg[0] return the big matrix, arg[1] return the list.
        #return (big_adj_mat)     
        self.adj_list = big_adj_mat

    def edges(self):
        """Creates an iterator with all graph edges."""
        edges = set()
        for node, neighbours in self.adj_list.items():
            for neighbour in neighbours:
                edge = frozenset([node, neighbour])
                if edge in edges:
                    continue
                edges.add(edge)
                yield node, neighbour

    def __len__(self):
        return len(self.elements)

    def __getitem__(self, position):
        return (self.elements[position], (self.x[position], self.y[position], self.z[position]), self.tresp[position], self.molname[position])

    
#Here is the function to convert self-defined graph object to the networkx style graph.
def to_networkx_graph(graph: MolGraph) -> nx.Graph:
    """
    Creates a NetworkX graph.
    Atomic elements and coordinates are added to the graph as node attributes 'element' and 'xyz" respectively.
    Modification: Added another attribute: tresp charges to the 'charge' slot
    
    Bond lengths are added to the graph as edge attribute 'length'
    """
    
    G = nx.Graph(graph.adj_list)
    node_attrs = {num: {'element': element, 'xyz': xyz, 'charge': tresp, 'molname': molname} for num, (element, xyz, tresp, molname) in enumerate(graph)}
    nx.set_node_attributes(G, node_attrs)
    edge_attrs = {edge: {'length': length} for edge, length in graph.bond_lengths.items()}
    nx.set_edge_attributes(G, edge_attrs)
    return G



#Here's a simple function to draw colored graph
def draw_color_graph(G):
    """
    draw a graph that color with different molecule name.
    """
    val_map = {
    "CLA": 1.0,
    "LUT": 0.5,
    "NEX": 0.2,
    "VIO": 0.0,
    }
    values = [val_map.get(node[:][1]["molname"], 0.25) for node in G.nodes(data=True)]
    
    nx.draw(G, cmap=plt.get_cmap('viridis'), node_color=values, with_labels=False, font_color='white')
    plt.show()
    return
    

#With graph generated from multiple molecules, here's a function to find the "cluster", or different components of the graph
def find_components(g):
    components = [g.subgraph(c).copy() for c in nx.connected_components(g)]
    for idx,g in enumerate(components,start=1):
        print(f"Component {idx}: Nodes: {g.nodes()} Edges: {g.edges()}")
        

#The last is to draw the largest N components.
def draw_nlargest_components(G):
    n=4
    largest_components = list(G.subgraph(c) for c in nx.connected_components(G))
    for index,component in enumerate(largest_components):
        draw_color_graph(component)
        plt.savefig('fig{}.pdf'.format(index))
        plt.show()


#Util functions that helps build the large adj matrix:
def add_matrix_corner(a,b):
    """
    takes in two square np array, return the larger array with b added to the corner of a.
    example:
    a = np.zeros([5,5])+1
    b = np.zeros([3,3])+2
    
    return:
    (np.pad(a, ((0,3)))+np.pad(b, (5,0)))
    """
    a_length = a.shape[0]
    b_length = b.shape[0]
    
    return (np.pad(a, ((0,b_length)))+np.pad(b, (a_length,0)))

def sum_adj_matrix(list_of_adj):
    """
    Takes in a list of adj matrix as numpy array.
    return the added up sum adj matrix
    """
    sum_adj = np.array([])
    for adj in list_of_adj:
        sum_adj = add_matrix_corner(sum_adj, adj)
    
    return sum_adj
    
    
def split_structure_connect(panda_df):
    """
    Given a pandas dataframe, spit out the data part and CONNECT part as two panda frames
    """
    CONNECT_START, CONNECT_END = 0,0
    for idx, row in panda_df.iterrows():
        if row[0] == "TER":
            CONNECT_START = idx
        if row[0] == "END":
            CONNECT_END = idx
    #print(CONNECT_START, CONNECT_END)    
    return (panda_df.iloc[0:CONNECT_START], panda_df.iloc[CONNECT_START+1:CONNECT_END])
    
        

