import numpy as np
import torch
from torch_geometric.data import DataLoader
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

from my_dataset_4cluster_2 import *
import time

import matplotlib.pyplot as plt

#import warnings
#warnings.filterwarnings('ignore')

class GCN_regression_Net(torch.nn.Module):
    def __init__(self, n_features, nhid1, nhid2, nhid3, batch_size):
        super(GCN_regression_Net, self).__init__()
        self.conv1 = GCNConv(n_features, nhid1)
        self.conv2 = GCNConv(nhid1, nhid2)
        self.conv3 = GCNConv(nhid2, nhid3)
        #self.bn = torch.nn.BatchNorm1d(n_nodes)
        #self.dropout = dropout
        #self.linear1 = torch.nn.Linear(nhid3, 1)    #used in previous patch, reduce nhid first
        self.linear1 = torch.nn.Linear(nhid3, 1)
        self.batch_size = batch_size
        #self.maxpool = torch.nn.MaxPool2d(4, stride = 4) # max pooling with square window of size=3, stride=2
        #no maxpooling because it will reduce the feature channel.
    
    
    def find_batch_length(self, batch_vec):
        """
        args:
            batch_vec is the list from GCN dataloader, essentially an array tells you which
            nodes belongs to which graph in the mini-batch.
            
            example:
            [0,0,0,0,0,0,0,1,1,1,1,1,1,2,2,2,2,2,2,3,3,3,3,3,3]
            
            the first 7 nodes belongs to the first graph etc...
            
        """
        
        graph_list = list(range(self.batch_size))
        graph_length_list = []
        
        num = 0
        
        for idx, element in enumerate(batch_vec):
            if element != graph_list[num]: # means the next graph starts from this idx
                graph_length_list.append(idx)
                num = num + 1
        
        out = []
        for idx, element in enumerate(graph_length_list):
            if idx == 0:
                out.append(element)
            else:
                out.append(element-graph_length_list[idx-1])
        
        #add the last chunk.
        out.append(len(batch_vec) - graph_length_list[-1])
        return out  # return a list of n_nodes in the mini-batch

        
    def forward(self, X, edge_index, edge_weight, batch_vec):
        x, edge_index, edge_weight, batch_vec = X, edge_index, edge_weight, batch_vec #X has shape[batch_size * n_node]

        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        
        x = F.dropout(x, training=self.training)
        
        x = self.conv2(x, edge_index, edge_weight)
        x = F.relu(x)
        
        x = self.conv3(x, edge_index, edge_weight)    # X in shape: [n_nodes * batch_size, nhid3]
        x = F.relu(x)
        
        x = F.dropout(x, training=self.training) 
        
        """
        #here is the method that reduce hidden dimension. 
        x = self.linear1(x)  #output shape in [batch_size * n_node, 1]
        
        #print("the size of the chunks are:")
        #print(self.find_batch_length(batch_vec))
        
        
        chunks = torch.split(x, (self.find_batch_length(batch_vec))) # a list of torch.tensor here.
        
        means = list(torch.mean(element) for element in chunks)
        x = torch.stack(means)
        """
        #here is the method that reduce the num_nodes.
        # need to do this in Module way. so it can be learned.
        chunks = torch.split(x, (self.find_batch_length(batch_vec))) # now chunks is a list of [n_nodes, nhnid3] with its length of batch_size
        
        temp = list(torch.sum(element, dim = 0) for element in chunks)  # now list is [1, nhid3], with its length of batch_size
        #print(np.array(element.cpu().detach()).shape)
                    
        x = torch.stack(temp) #output shape in [10, 128] 
         
        x = self.linear1(x) #output shape in [10, 1]  
        x = torch.squeeze(x)
        #print("the output layer shape is:")
        #print(x.shape)
        
        
        #x = torch.squeeze(x)    #output shape in [batch_size * n_node]
        #print(type(x))
        #x = torch.mean(x)
        #x = torch.reshape(x, (self.batch_size, -1)) # output shape in [batch_size, n_node]
        return x
        
class RMSELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss()
        
    def forward(self,yhat,y):
        return torch.sqrt(self.mse(yhat,y))

    
#########################   added an ATTENTION layer version ##################
class GCN_regression_attn_Net(torch.nn.Module):
    def __init__(self, n_features, nhid1, nhid2, nhid3, batch_size):
        super(GCN_regression_attn_Net, self).__init__()
        self.conv1 = GCNConv(n_features, nhid1)
        self.conv2 = GCNConv(nhid1, nhid2)
        self.conv3 = GCNConv(nhid2, nhid3)
        #self.bn = torch.nn.BatchNorm1d(n_nodes)
        #self.dropout = dropout
        #self.linear1 = torch.nn.Linear(nhid3, 1)    #used in previous patch, reduce nhid first
        self.linear1 = torch.nn.Linear(nhid3, 1)
        self.linear2 = torch.nn.Linear(508, 1)
        self.batch_size = batch_size
        #self.maxpool = torch.nn.MaxPool2d(4, stride = 4) # max pooling with square window of size=3, stride=2
        #no maxpooling because it will reduce the feature channel.
        self.multihead_attn = torch.nn.MultiheadAttention(nhid3, 16) #(embed_dim, num_heads)
    
    
    def find_batch_length(self, batch_vec):
        """
        args:
            batch_vec is the list from GCN dataloader, essentially an array tells you which
            nodes belongs to which graph in the mini-batch.
            
            example:
            [0,0,0,0,0,0,0,1,1,1,1,1,1,2,2,2,2,2,2,3,3,3,3,3,3]
            
            the first 7 nodes belongs to the first graph etc...
            
        """
        
        graph_list = list(range(self.batch_size))
        graph_length_list = []
        
        num = 0
        
        for idx, element in enumerate(batch_vec):
            if element != graph_list[num]: # means the next graph starts from this idx
                graph_length_list.append(idx)
                num = num + 1
        
        out = []
        for idx, element in enumerate(graph_length_list):
            if idx == 0:
                out.append(element)
            else:
                out.append(element-graph_length_list[idx-1])
        
        #add the last chunk.
        out.append(len(batch_vec) - graph_length_list[-1])
        return out  # return a list of n_nodes in the mini-batch

        
    def forward(self, X, edge_index, edge_weight, batch_vec):
        x, edge_index, edge_weight, batch_vec = X, edge_index, edge_weight, batch_vec #X has shape[batch_size * n_node]

        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        
        x = F.dropout(x, training=self.training)
        
        x = self.conv2(x, edge_index, edge_weight)
        x = F.relu(x)
        
        x = self.conv3(x, edge_index, edge_weight)    # X in shape: [n_nodes * batch_size, nhid3]
        x = F.relu(x)
        
        x = F.dropout(x, training=self.training) 
        
        """
        #here is the method that reduce hidden dimension. 
        x = self.linear1(x)  #output shape in [batch_size * n_node, 1]
        
        #print("the size of the chunks are:")
        #print(self.find_batch_length(batch_vec))
        
        
        chunks = torch.split(x, (self.find_batch_length(batch_vec))) # a list of torch.tensor here.
        
        means = list(torch.mean(element) for element in chunks)
        x = torch.stack(means)
        """
        #here is the method that reduce the num_nodes.
        # need to do this in Module way. so it can be learned.
        chunks = torch.split(x, (self.find_batch_length(batch_vec))) # now chunks is a list of [n_nodes, nhnid3] with its length of batch_size
        
        #temp = list(torch.sum(element, dim = 0) for element in chunks)  # now list is [1, nhid3], with its length of batch_size
        #print(np.array(element.cpu().detach()).shape)
                    
        x = torch.stack(chunks) #output shape in [16, 508, 128]
        #print("before the attn")
        #print(x.shape)
        
        attn_output, _ = self.multihead_attn(x, x, x) #output shape in [16, 508, 128] # we can use linear layer to reduce it to [16] but then?
        #print("after the attn")
        #print(attn_output.shape)
        
        x = self.linear1(attn_output) #output shape in [16, 508, 1] 
        #print("after the lin1")
        #print(x.shape)
        
        x = torch.squeeze(x) #output shape in [16, 508]
        #print(x.shape)
        x = self.linear2(x) #output shape in [16, 1]
        #print("after the lin2")
        #print(x.shape)
        x = torch.squeeze(x)  #output shape same as batch_size
        #print("the output layer shape is:")
        #print(x.shape)
        
        
        #x = torch.squeeze(x)    #output shape in [batch_size * n_node]
        #print(type(x))
        #x = torch.mean(x)
        #x = torch.reshape(x, (self.batch_size, -1)) # output shape in [batch_size, n_node]
        
        return x
    