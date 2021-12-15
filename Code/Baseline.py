#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 19:50:38 2021

@author: saul
"""

import torch
from torch import nn
from torch.autograd import grad

activations = {
    'tanh': nn.Tanh,
    'relu': nn.ReLU,
    'elu': nn.ELU,
    'softplus': nn.Softplus,
    'sigmoid': nn.Sigmoid,
    'leaky_relu': nn.LeakyReLU,
    'selu': nn.SELU,
    'identity': nn.Identity
}

def init(input_size, output_size, n_hidden_layers, n_neurons, activation, ln):
        activation=activations[activation]
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
class FF_Network(nn.Module):
    def __init__(self, input_size, output_size, n_layers, n_neurons, device, activation=None, ln=False):
        super().__init__()
        model=init(input_size, output_size, n_layers, n_neurons, activation, ln)
        # for i in range(0,len(model)):
        #     self.weights_init(model[i])
        self.ff_nn = torch.nn.Sequential(*model)
    # def weights_init(self,model):
    #     classname = model.__class__.__name__
    #     if classname.find('Linear') != -1:
    #         # nn.init.constant_(model.bias, 0)
    #         # nn.init.normal_(model.bias)
    #         # nn.init.orthogonal_(model.weight,  gain=nn.init.calculate_gain('tanh'))
    #         # torch.nn.init.uniform_(model.weight, -lim, lim)
    #         # torch.nn.init.xavier_normal_(model.weight)


    def forward(self,qqd):
        return self.ff_nn(qqd)  
    
class LNN(nn.Module):
    def __init__(self, input_size, output_size, n_layers, n_neurons, h, activation=None, custom_init=None, ln=True, rk4_step=False):
        self.h=h
        self.custom_init=custom_init
        self.input_size=input_size
        self.n_layers=n_layers
        self.n_neurons=n_neurons
        self.rk4_step=rk4_step
        
        super().__init__()
        model=init(input_size, output_size, n_layers, n_neurons, activation, ln)
    
        if self.custom_init:
            step=2
            if ln:
                step=3
            for i in range(0,len(model), step):
                self.weights_init(model[i],i//step)
                nn.init.constant_(model[i].bias, 0)
        self.ff_nn = torch.nn.Sequential(*model)
        
  
    def weights_init(self,model, i):
        classname = model.__class__.__name__
        if classname.find('Linear') != -1:
            if i==0:
                nn.init.normal_(model.weight, 0, 2.2/torch.sqrt(torch.tensor(self.n_neurons,dtype=float)))
            elif i==(self.n_layers-1):
                nn.init.normal_(model.weight, 0, self.n_neurons/torch.sqrt(torch.tensor(self.n_neurons,dtype=float)))
            else :
                nn.init.normal_(model.weight, 0, (0.58*i)/torch.sqrt(torch.tensor(self.n_neurons,dtype=float)))
                
    def forward(self, x):
        with torch.set_grad_enabled(True):
            qqd = x.requires_grad_(True)
            if self.rk4_step:
                out=self._rk4_step(qqd)
            else:
                out=self.euler_lagrange(qqd)
        return out
                
   
    def _lagrangian(self, qqd):
        return self.ff_nn(qqd) 
    
    def euler_lagrange(self,qqd):
        self.n = n = qqd.shape[1]//2
        L = self._lagrangian(qqd).sum()
        J = grad(L, qqd, create_graph=True)[0] ;
        DL_q, DL_qd = J[:,:n], J[:,n:]
        DDL_qd = []
        for i in range(n):
            J_qd_i = DL_qd[:,i][:,None]
            H_i = grad(J_qd_i.sum(), qqd, create_graph=True)[0][:,:,None]
            DDL_qd.append(H_i)
        DDL_qd = torch.cat(DDL_qd, 2)
        DDL_qqd, DDL_qdqd = DDL_qd[:,:n,:], DDL_qd[:,n:,:]
        T = torch.einsum('ijk, ij -> ik', DDL_qqd, qqd[:,n:])
        qdd = torch.einsum('ijk, ij -> ik', DDL_qdqd.inverse(), DL_q - T)
        return qdd
    
    
    def _rk4_step(self, qqd):
        k1 = self.h * self.euler_lagrange(qqd)
        k2 = self.h * self.euler_lagrange(qqd + k1/2)
        k3 = self.h * self.euler_lagrange(qqd + k2/2)
        k4 = self.h *self.euler_lagrange(qqd + k3)
        return qqd + 1/6 * (k1 + 2 * k2 + 2 * k3 + k4)
    
class PINN(nn.Module):
    """
    Input shape: torch.tensor([[q1_past, q2_past, q1, a2, q1_next, q2_next]]): 
        N x 3*d - Where N is the mini batch size and d the number of degrees of freedom 
    """
    def __init__(self, input_size, n_layers, n_neurons, device, activation=None, ln=False):
        super().__init__()
        self.d = int(input_size/(2*3))
        self.d_input = int(self.d*2)
        output_size = 1
        model=init(self.d_input, output_size, n_layers, n_neurons, activation, ln)
        self.lnn = torch.nn.Sequential(*model)
        model=init(int(self.d*5), self.d, n_layers, n_neurons, activation, ln)
        self.q = torch.nn.Sequential(*model)
  
    def forward(self, x):
        with torch.set_grad_enabled(True):
            q_past, qk,_, u_past, u, u_next =  torch.split(x,self.d,1)
            q  = qk.requires_grad_(True)
            q_next = self._position(q_past, q, u_past, u, u_next)
            qq_next = torch.cat((q,q_next),1)
            qq_past = torch.cat((q_past, q),1)
            L2 = self._lagrangian(qq_past)
            L1 = self._lagrangian(qq_next)
            f = L1 + L2
            DEL = grad(f.sum(), q, create_graph=True)[0]
            out = DEL + 0.25*0.01*(u_past + u) + 0.25*0.01*(u + u_next)
            # print(q_next)
            # print(out)
            return out, q_next

    def _lagrangian(self, q):
        return self.lnn(q)
        
    def _position(self, q_past, q, u_past, u, u_next):
        input = torch.cat((q_past,q, u_past, u, u_next ),1)
        return self.q(input)
    
    def loss(self, output, target):
        N = len(output[0])
        loss = (output[0]**2)/N + ((output[1] - target)**2)/N
        
        return loss.sum() 
    
class HNN(nn.Module):
   def __init__(self, input_size, output_size, n_layers, n_neurons, activation=None, ln=False):
        super().__init__()
        model=init(input_size, output_size, n_layers, n_neurons, activation, ln)
        self.hnn = torch.nn.Sequential(*model)
   def forward(self, x):
        with torch.set_grad_enabled(True):
            self.n = x.shape[1]//2
            x = x.requires_grad_(True)
            gradH = torch.autograd.grad(self._hamiltonean(x).sum(), x, allow_unused=False, create_graph=True)[0]
        return torch.cat([gradH[:,self.n:], -gradH[:,:self.n]], 1).to(x)

   def _hamiltonean(self, qqd):
       return self.hnn(qqd)
   
   
class DELAN(nn.Module):
    def __init__(self, input_size, n_layers, n_neurons, device, mask, activation=None, ln=False):
        super().__init__()
        self.input_dim = input_size//3
        self.device = device
        self.mask = mask.to(self.device)
        self.activation = activation
        self.n_neurons = n_neurons
        output_size=None
        self.model = init(self.input_dim, output_size, n_layers, n_neurons, activation, ln=False)    
        self.delan = torch.nn.Sequential(*self.model)
        self.neg_slope=0.01
        self.diagonal_layer = nn.functional.softplus
        
        # gravity layer
        self.fc2 = nn.Linear(self.n_neurons, self.input_dim) 
    
        # ld layer
        self.fc3 = nn.Linear(self.n_neurons, self.input_dim)
    
        # lo layer
        self.d_n_terms=((self.input_dim**2)-self.input_dim)/2
        self.fc4 = nn.Linear(self.n_neurons, int(self.d_n_terms))
        
    def get_analytical_derivatives(self, hi, activation):
         dact_dhi ={
             'leaky_relu' : lambda: torch.where(hi > 0, torch.ones(hi.shape,device=self.device), self.neg_slope * torch.ones(hi.shape,device=self.device)),
             'relu' : lambda: torch.where(hi > 0, torch.ones(hi.shape,device=self.device), torch.zeros(hi.shape,device=self.device)),
             'softplus' : lambda: torch.sigmoid(hi),
             'elu' : lambda: torch.where(hi > 0, torch.ones(hi.shape,device=self.device), hi + 1)
             }
         return dact_dhi[activation]()
     
    def embed_angle(self, q):
        theta = torch.masked_select(q, self.mask)
        x = torch.cos(theta)
        y = torch.sin(theta)
        p = torch.masked_select(q, ~self.mask)
        cartesian = torch.stack([x,y, p], axis = -1)
        return cartesian
    
    def reshape(self, ld, lo, q_dot, dld_dhi, dlo_dhi):
        # Get L, dL matrices without inplace operations
        n=self.n
        d=self.d
        dld_dqi = dld_dhi.permute(0,2,1).view(n,d,d,1)
        dlo_dqi = dlo_dhi.permute(0,2,1).view(n,d,-1,1)

        dld_dt = dld_dhi @ q_dot.view(n,d,1)
        dlo_dt = dlo_dhi @ q_dot.view(n,d,1)
        L = []
        dL_dt = []
        dL_dqi = []
        zeros = torch.zeros_like(ld)
        zeros_2 = torch.zeros_like(dld_dqi)
        lo_start = 0
        lo_end = d - 1
        for i in range(d):
            l = torch.cat((zeros[:, :i].view(n, -1), ld[:, i].view(-1, 1), lo[:, lo_start:lo_end]), dim=1)
            dl_dt = torch.cat((zeros[:, :i].view(n, -1), dld_dt[:, i].view(-1, 1),
                               dlo_dt[:, lo_start:lo_end].view(n, -1)), dim=1)
        
            dl_dqi = torch.cat((zeros_2[:, :, :i].view(n, d, -1), dld_dqi[:, :, i].view(n, -1, 1),
                                dlo_dqi[:, :, lo_start:lo_end].view(n, d, -1)), dim=2)
        
            lo_start = lo_start + lo_end
            lo_end = lo_end + d - 2 - i
            L.append(l)
            dL_dt.append(dl_dt)
            dL_dqi.append(dl_dqi)
        
        L = torch.stack(L, dim=2)
        dL_dt = torch.stack(dL_dt, dim=2)
        
        dL_dqi = torch.stack(dL_dqi, dim=3).permute(0, 2, 3, 1)
        
        return L, dL_dt, dL_dqi
    
    def inertia_matrix(self, L):
        epsilon = 1e-9   #small number to ensure positive definiteness of H

        return L @ L.transpose(1, 2) + epsilon * torch.eye(self.d, device=self.device)
    def coriolis_matrix(self, L, dL_dqi, q_dot, dL_dt):
        d=self.d
        n=self.n
        # Time derivative of Mass Matrix
        dH_dt = L @ dL_dt.permute(0,2,1) + dL_dt @ L.permute(0,2,1)
        quadratic_term = []
        for i in range(d):
            qterm = q_dot.view(n, 1, d) @ (dL_dqi[:, :, :, i] @ L.transpose(1, 2) +
                                           L @ dL_dqi[:, :, :, i].transpose(1, 2)) @ q_dot.view(n, d, 1)
            quadratic_term.append(qterm)

        quadratic_term = torch.stack(quadratic_term, dim=1)
        
        return dH_dt @ q_dot.view(n,d,1) - 0.5 * quadratic_term.view(n,d,1)
    
    def matrix_layers(self, hi, dhii_dhi):
        # Gravity torque
        g = self.fc2(hi)
    
        # ld is vector of diagonal L terms, lo is vector of off-diagonal L terms
        h3 = self.fc3(hi)
        #Positive activation function to guarantee positive definitess of H
        ld = self.diagonal_layer(h3)
        lo = self.fc4(hi)
        
        #Analytical derivatives of matrix layers
        dld_dhi = torch.diag_embed(self.get_analytical_derivatives(h3,'softplus')) @ self.fc3.weight
        dlo_dhi = self.fc4.weight
        dld_dhi =dld_dhi @ dhii_dhi
        dlo_dhi = dlo_dhi @ dhii_dhi
        
        return g, ld, lo, dld_dhi, dlo_dhi
  
    def forward(self, x):
        if (x.shape[1]%3) != 0:
            self.d = d = x.shape[1] // 2
            q, q_dot = torch.split(x,[d,d], dim = 1)
            forced = False
        else:
            self.d = d = x.shape[1] // 3
            q, q_dot, tau = torch.split(x,[d,d,d], dim = 1)
            forced = True
        self.n = n = x.shape[0]
        self.embed_angle(q)
        #common layers analytical derivatives and forward:
        hi=q
        dhii_dhi=torch.eye(d, device = self.device)
        dhii_dhi = dhii_dhi.reshape((1, d, d))
        dhii_dhi = dhii_dhi.repeat(n, 1, 1)
        for layer in self.delan:
            if isinstance(layer, nn.Linear):
                affine=layer
                ai = affine(hi)
            else:
                hi = layer(ai)
                dact_dfci=self.get_analytical_derivatives(hi, self.activation)
                dhii_dhi = torch.diag_embed(dact_dfci) @ affine.weight @ dhii_dhi
        
        #Analytical derivatives and matrices    
        g, ld, lo, dld_dhi, dlo_dhi = self.matrix_layers(hi, dhii_dhi)
        
    
        #Get reshaped terms
        L, dL_dt, dL_dqi=self.reshape(ld, lo, q_dot, dld_dhi, dlo_dhi)
        
        #Get Inertia Matrix
        H = self.inertia_matrix(L)

        #Get coriolis
        c=self.coriolis_matrix(L, dL_dqi, q_dot, dL_dt)
        
        #Inverse Euler_lagrange
        if forced:
            q_ddot = torch.solve(tau.view(n,d,1) - c - g.view(n,d,1), H)[0]
        else:
            q_ddot = torch.solve(-c - g.view(n,d,1), H)[0]
        # The loss layer will be applied outside Network class
        return q_ddot.squeeze()