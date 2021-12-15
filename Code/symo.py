import torch
from torch import nn
from torch.autograd import grad
from abc import ABC, abstractmethod
from Code.Rigid_Body import Rigid_Body, MLP
from Code.del_module import DEL_Module


class SyMo_T(nn.Module, DEL_Module):
    def __init__(self, n_angles, n_layers, n_neurons, h, activation, embedding=False, ln=False):
        super().__init__()
        self.n_neurons = n_neurons
        self.h = h
        self.embedding = embedding  
        H_params = int(n_angles*2)
        output_size = None #multi-headed
        FF = MLP(H_params, output_size, n_layers, n_neurons, activation, ln)    
        RB = Rigid_Body(FF, n_angles, n_neurons)
        self.nn = RB
        self.n_angles = n_angles

    def lagrangian(self, q, qd):
        new_q = self.angle_embedding(q)            
        H, V = self.rigid_body(new_q)
        T = self._kinetic_energy(H, qd)    
        return (T - V)*self.h
    
    def rigid_body(self, q):
        return self.nn(q)

    def angle_embedding(self, q):
        x = torch.cos(q)
        y = torch.sin(q)
        new_q = torch.cat((x,y), 1)
        return new_q  
    
    def get_matrices(self, x):
        self.d = x.shape[1] // 2 #number of degrees of freedom
        h = self.h
        q_next, qdot =  torch.split(x, self.d, 1)  

        new_q = self.angle_embedding(q_next)
        H, V = self.nn(new_q)
        T = self._kinetic_energy(H, qdot)  
        return H.detach(), V.detach(), T.detach()
    
    def forward(self, x):
        q_past, q, q_next, u_past, u, u_next =  torch.split(x, self.n_angles, 1)
        args = [q_past, q, u_past, u, u_next, self.h]
        #Newton Layer
        DEL = self._DEL(q_next, *args)
        return DEL

    #only used for newton layer
    def jacobian(self, x):
        q_past, q, q_next, u_past, u, u_next =  torch.split(x, self.n_angles, 1)
        args = [q_past, q, u_past, u, u_next, self.h]
        return self._Jacobian(q_next, *args)
    
    def momentum(self, x):
        q_past, q = torch.split(x, self.n_angles, 1)
        return self._momentums(q_past, q, self.h)
        
    def implicit_loss(self, output, target):
        criterion = nn.MSELoss()
        return criterion(output, target)

    def loss(self, output, target):
        return torch.mean(output**2)
    
class SyMo_RT(nn.Module, DEL_Module):
    def __init__(self, m, n, n_layers, n_neurons, h, activation, embedding=False, ln=False):
        super().__init__()
        self.n_neurons = n_neurons
        self.h = h
        self.d_f = m + n #n of translational + n of rotational coordinates
        output_size = None
        self.m = m
        self.n = n
        self.embedding = embedding  
        H_params = m + int(n*2) #params with embedding
        output_size = None #multi-headed
        FF = MLP(H_params, output_size, n_layers, n_neurons, activation, ln)    

        RB = Rigid_Body(FF, self.d_f, n_neurons)
        self.nn = RB
    
    def angle_embedding(self, q):
        r = q[:, :self.m]
        theta = q[:, self.m:] 
        x = torch.cos(theta)
        y = torch.sin(theta)
        new_q = torch.cat((r, x, y), 1)
        return new_q 

    def lagrangian(self, q, qd):
        q = self.angle_embedding(q)            
        H, V = self.nn(q)
        T = self._kinetic_energy(H, qd)    
        return (T - V)*self.h
    
    def get_matrices(self, x):
        self.d = x.shape[1] // 2 #number of degrees of freedom
        h = self.h
        q_next, qdot =  torch.split(x, self.d, 1)  

        new_q = self.angle_embedding(q_next)
        H, V = self.nn(new_q)
        T = self._kinetic_energy(H, qdot)  
        return H.detach(), V.detach(), T.detach()
    
    def forward(self, x):
        q_past, q, q_next, u_past, u, u_next =  torch.split(x, self.d_f, 1)
        args = [q_past, q, u_past, u, u_next, self.h]
        #Newton Layer
        DEL = self._DEL(q_next, *args)
        return DEL

    #only used for newton layer
    def jacobian(self, x):
        q_past, q, q_next, u_past, u, u_next =  torch.split(x, self.d_f, 1)
        args = [q_past, q, u_past, u, u_next, self.h]
        return self._Jacobian(q_next, *args)
        
    def implicit_loss(self, output, target):
        criterion = nn.MSELoss()
        return criterion(output, target)
    
    def loss(self, output, target):
        return torch.mean(output**2)