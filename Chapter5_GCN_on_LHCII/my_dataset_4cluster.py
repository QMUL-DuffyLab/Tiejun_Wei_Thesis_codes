#here we build a dataloader, a tool/interface that allow us to access data in batches
#plus, it's parallenlizable and faster in large dataset.
#What it actaully do:
# load 4 cluster data at one time.
# generate the graph object in networkX style (Transformation)
# Padding zeros to data with different shape(e.g. x1=[508, 6]; x2=[506, 6]); this also will help mutant model later on.
# batching; batch normalization.

import torch
from xyz_tresp_2networkx import *
import networkx as nx
from torch_geometric import utils
from torch_geometric.data import Data
import os
from sklearn.preprocessing import *

class My_dataset(torch.utils.data.Dataset):
    "Characterizes our own dataset"
    def __init__(self, PATH):
        """
        args:
            PATH: the main directory containing all 18 pigments folders
            transform: a callable, transformation to be applied on the sample.
        """
        
        #self.labels = labels
        self.PATH = PATH
        self.dir_list = ["CLA610/", "CLA611/", "CLA612/", "LUT620/"]
        self.label_path = "./all_lifetimes.csv"
        
    def __len__(self):
        "denotes the total number of samples"
        #we take the first folder here as example and return its number of contents.
        sample_dir = os.listdir(self.PATH)[0]
        #print("The length of this dataset is: %d" %len(os.listdir(self.PATH + sample_dir + "/")))
        return len(os.listdir(self.PATH + sample_dir + "/"))
    
    def __cat_dim__(self, key, item):
        
        #or should I use:
        if key in ('X', 'y', 'index'):
        #if key == 'y':
            return None
        else: 
            return super().__cat_dim__(key, item)
    
    def __getitem__(self, index):
        "generates one sample of data"
        
        work_path = []
        for element in self.dir_list:
            work_path.append(self.PATH + element + "frame" + str(index+1) + ".csv")
        
        
        #print("Loading data")
        #for element in work_path:
        #    print(element)
        
        
        #load the data and transform it to torch.data object
        mg = MolGraph()
        mg.read_4callumcsv(work_path)
        data = utils.from_networkx(to_networkx_graph(mg))
        
        label_list = np.loadtxt(self.label_path)
        
        out = assemble_data_point(data, label_list[index])
        
        #print(index)
        return out


#Until this point we defined the dataset class.
#Next we need introduce the dataloader class, which has following important arguements:




#Utility tools here:
def molname_mapping(string):
    """
    input the string, output the integer
    """
    if string == 'CLA':
        out = 6
    elif string == 'CHL':
        out = 1
    elif string == 'NEX':
        out = 2
    elif string == 'VIO':
        out = 3
    elif string == 'LUT':
        out = 4
    elif string == 'ZEA':
        out = 5
    else:
        print("something went wrong when mapping the molname")
        print(string)
    return (out)
    

def element_mapping(string):
    """
    input the string, output the integer
    """
    if string == 'C':
        out = 5
    elif string == 'H':
        out = 1
    elif string == 'O':
        out = 2
    elif string == 'N':
        out = 3
    elif string == 'Mg':
        out = 4
    else:
        print("something went wrong when mapping the element")
        print(string)
    return (out)

def assemble_data_point(data_point, label, norm = True):
    """
    input is the product of torch function: utils.from_networkx(); this is a torch.data.Data type object.
    the second input is the corresponding lifetime for this datapoint
    
    however we need clean up the features in the data_point and output the cleaned datapoint such it has:
    Data(x=[508,6], edge_index, edge_attr, y)
    
    for details go to "meet_with_JW.png"
    
    we have to move the "dimension expansion" before cat tensor.
    
    
    added the pre-processing: normalization on the xyz, tresp charge. no change on the rest features.lifetime(y)
    
    """
    tensor_molname = torch.as_tensor(list(map(molname_mapping, data_point.molname)))
    tensor_element = torch.as_tensor(list(map(element_mapping, data_point.element)))
    tensor_charge = torch.as_tensor(data_point.charge)
    tensor_length = torch.as_tensor(data_point.length)
    
    #first we need creat new dimension in order to concatenate.
    if tensor_molname.dim()==1:
        tensor_molname.unsqueeze_(1)
        #print(tensor_molname.dim())
        tensor_element.unsqueeze_(1)
        tensor_charge.unsqueeze_(1)
        tensor_length.unsqueeze_(1)
        
    #normalization here.
    if norm == True:
        #print(type(data_point.xyz))
        #print(data_point.xyz.shape)
        x = data_point.xyz[:, 0].double()
        y = data_point.xyz[:, 1].double()
        z = data_point.xyz[:, 2].double()
        #print("x before the norm:")
        #print(x)
        x_norm = torch.tensor(normalize(x[:,np.newaxis], axis=0), dtype = torch.double)
        y_norm = torch.tensor(normalize(y[:,np.newaxis], axis=0), dtype = torch.double)
        z_norm = torch.tensor(normalize(z[:,np.newaxis], axis=0), dtype = torch.double)
        #print("x after the norm:")
        #print(type(x_norm))
        
        tensor_charge_norm = torch.as_tensor(normalize(tensor_charge)).double()
        #label_norm = torch.as_tensor(label)
        
    #then we map the molname into integers based on function 
    
    """
    for element in (x_norm, y_norm, z_norm, tensor_charge_norm, tensor_element, tensor_molname):
        print(element.shape)"""
    
    out = Data(X= torch.cat((x_norm, y_norm, z_norm, tensor_charge_norm, tensor_element, tensor_molname), 1), 
            edge_index = data_point.edge_index, 
            edge_attr = tensor_length, 
            y = label)
    return (out)

"""
    def feature_norm(datapoint):
        #input the torch.data object
        for element in datapoint:
            print("before the normalization:")
            print(element)
            element = normalize(element, axis = 0)
            print("after the normalization:")
            print(element)
        return"""