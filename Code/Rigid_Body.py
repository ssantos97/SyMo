

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 20:27:25 2021

@author: saul
"""

import torch
from torch import nn
from Code.Utils import non_linearity
import numpy as np

class MLP(nn.Module):
    """
    Returns a Multi Layer Perceptron
    """
    def __init__(self, input_size, output_size, n_hidden_layers, n_neurons, activation, ln):
        super().__init__()
        model = self.init(input_size, output_size, n_hidden_layers, n_neurons, activation, ln)
        
        for i in range(0,len(model)):
            self.weight_init(model[i], activation)
        
        self.MLP = torch.nn.Sequential(*model)
    
    def init(self, input_size, output_size, n_hidden_layers, n_neurons, activation, ln):
        activation=non_linearity(activation)
        model = [torch.nn.Linear(input_size, n_neurons)]
        model.append(activation())
        if ln==True:
            model.append(torch.nn.LayerNorm(n_neurons))
        
        for i in range(n_hidden_layers-1):
            model.append(torch.nn.Linear(n_neurons, n_neurons))
            model.append(activation())
            if ln:
                model.append(torch.nn.LayerNorm(n_neurons))
        if output_size is not None:
            model.append(torch.nn.Linear(n_neurons, output_size))
        
        return model

    def weight_init(self,model, activation):
        classname = model.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.constant_(model.bias, 0)
            nn.init.xavier_uniform_(model.weight, gain=nn.init.calculate_gain(activation))


    def forward(self, q):
        return self.MLP(q)
    
class Rigid_Body(nn.Module):
    
    def __init__(self, FF, dim, neurons, B = None):
        super().__init__()
        self.MLP = FF
        self.dim = dim
        self.neurons = neurons
        self.B = B
        #ld layer
        self.fc_ld = nn.Linear(self.neurons, self.dim)
        self.diagonal_layer = nn.functional.softplus
        # self.B = nn.Linear(self.neurons, 1)
        # lo layer
        d_n_terms=((self.dim**2)-self.dim)/2
        if d_n_terms != 0:
            self.fc_lo = nn.Linear(self.neurons, int(d_n_terms))
        else:
            self.fc_lo = None
       
        if self.B is not None:
            self.B = nn.Linear(self.neurons, self.dim)
        self.fc_g = nn.Linear(self.neurons, 1)
        
    def Cholesky_decomposition(self, Ld, Lo):
        """
        Assembled a lower triangular matrix from it's diagonal and off-diagonal elements
        :param Lo: Off diagonal elements of lower triangular matrix
        :param Ld: Diagonal elements of lower triangular matrix
        :return: Lower triangular matrix L
        """
    
        L = torch.diag_embed(Ld)
        
        ind = np.tril_indices(self.dim, k=-1)
        flat_ind = np.ravel_multi_index(ind, (self.dim, self.dim))
        L = torch.flatten(L, start_dim=1)
        L[:, flat_ind] = Lo
        L = torch.reshape(L, (self.batch, self.dim, self.dim))
        return L    

    def inertia_matrix(self, hi):
        # self.fc_ld.weight.register_hook(lambda x: print('grad accumulated in fc1'))
        Ld = self.diagonal_layer(self.fc_ld(hi)) #ensures positive diagonal elements
        if self.dim == 1:
            return Ld*Ld + 1e-9
        else: 
            Lo = self.fc_lo(hi)
            L = self.Cholesky_decomposition(Ld, Lo)
            H = torch.bmm(L, L.permute(0, 2, 1))
            H[:, 0, 0] = H[:, 0, 0] + 1e-9
            H[:, 1, 1] = H[:, 1, 1] + 1e-9
            # print(torch.mean(H[:1,1]))
            return H
    
    def potential(self, hi):
        V =  self.fc_g(hi)
        return V
    
    def input_matrix(self, hi):
        B = self.B(hi)
        if self.dim > 1:
            input_matrix = torch.diag(B)
        return input_matrix
    
    def forward(self, q):
        self.batch = q.shape[0]
        with torch.set_grad_enabled(True):
            hi = self.MLP(q) 
            V = self.potential(hi)
            if self.fc_lo is not None:
                H = self.inertia_matrix(hi)
            else:
                H = torch.unsqueeze(self.diagonal_layer(self.fc_ld(hi)),1)
                # H = torch.ones_like(V)*0.33333333
                # H = torch.unsqueeze(H,1)
        return H, V