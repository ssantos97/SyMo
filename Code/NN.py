import torch
from torch import nn
from torch.autograd import grad
from abc import ABC, abstractmethod
from Code.Rigid_Body import Rigid_Body, MLP
from torchdiffeq import odeint_adjoint as odeint

class NODE_RT(nn.Module):
    """
    Cartpole implementation - Rotational and translational coordinates    
    Manifold here hybrid R^m x T^n (for n_angles = 1) and consequently the number of 
    input paramters of the inertia matrix will be twice the number of angles: 
        r, theta ---> r, (cos(theta), sin(theta))
        
    Data structure: tensor[mxn] where m is the bacth size and n the number of elements of the state-space
    """
    def __init__(self, m, n, n_layers, n_neurons, activation, u_index, embedding=True, ln=False):
        super().__init__()
        self.d_f = m + n
        self.u_index = u_index
        params = m+int(2*n) + self.d_f + 1 #+1 because there's only one control (underactuated)
        output_size = self.d_f
        self.embedding = embedding
        self.m = m
        self.n = n
        self.FF = MLP(params, output_size, n_layers, n_neurons, activation, ln)    

    def angle_embedding(self, q):
        r = q[:,:self.m]
        theta = q[:,self.m:] 
        x = torch.cos(theta)
        y = torch.sin(theta)
        new_q = torch.cat((r,x,y), 1)
        return new_q  
  
    def forward(self, t, x):
        q, q_dot, tau = torch.split(x, self.d_f, 1)
        if self.embedding:
            q = self.angle_embedding(q)
        
        x = torch.cat((q, q_dot, tau[:, self.u_index][:, None]), dim=1)
        ddx_dt = self.FF(x)
        dx_dt = q_dot
        du_dt = torch.zeros_like(dx_dt)
        dx_dt = torch.cat((dx_dt, ddx_dt, du_dt), dim=1)
        return dx_dt
    

    
    def loss(self, output, target):
        criterion = nn.MSELoss()
        return criterion(output, target)
    
    
    
class NODE_T(nn.Module):
    """
    Cartpole implementation - Rotational and translational coordinates    
    Manifold here hybrid R^m x T^n (for n_angles = 1) and consequently the number of 
    input paramters of the inertia matrix will be twice the number of angles: 
        r, theta ---> r, (cos(theta), sin(theta))
        
    Data structure: tensor[mxn] where m is the bacth size and n the number of elements of the state-space
    """
    def __init__(self, n_angles, n_layers, n_neurons, activation, u_index, embedding=True, ln=False):
        super().__init__()
        self.d_f = n_angles
        self.embedding = embedding
        self.u_index = u_index
        params = int(n_angles*2 + self.d_f + 1) #+1 because there's only one control (underactuated)
        output_size = self.d_f
        self.FF = MLP(params, output_size, n_layers, n_neurons, activation, ln)    

    def angle_embedding(self, q):
        x = torch.cos(q)
        y = torch.sin(q)
        new_q = torch.cat((x,y), 1)
        return new_q  
  
    def forward(self, t, x):
        q, q_dot, tau = torch.split(x, self.d_f, 1)
        if self.embedding:
            new_q = self.angle_embedding(q)
        
        x = torch.cat((new_q, q_dot, tau[:, self.u_index][:, None]), dim=1)
        ddx_dt = self.FF(x)
        dx_dt = q_dot
        du_dt = torch.zeros_like(dx_dt)
        dx_dt = torch.cat((dx_dt, ddx_dt, du_dt), dim=1)
        return dx_dt
    
    def loss(self, output, target):
        criterion = nn.MSELoss()
        return criterion(output, target)
        
    
class Euler_Lagrange_Module(object):   
    """
    Deep Lagrangian Neural Networks Implementation
    Paper: Deep Lagrangian Networks: Using Physics as Model Prior for Deep Learning, Michael Lutter, Christian Ritter & Jan Peters ∗
    Learning in enbedded data to avoid traditional difficulties with learning angle discontinuities.
    """
    
    def _get_data(self, x):
        self.n = x.shape[0]
        self.d = d = x.shape[1] // 3
        q, q_dot, tau = torch.split(x, [d,d,d], dim = 1)
        return q, q_dot, tau
    
    def _coriolis(self, H, q, q_dot):
        Hv = H @ q_dot.unsqueeze(2)

        KE = 0.5 * q_dot.unsqueeze(1) @ Hv

        KE_field = torch.autograd.grad(KE.sum(), q, retain_graph=True, create_graph=True)[0]

        Hv_field = torch.stack([
            torch.autograd.grad(Hv[:, i].sum(), q, retain_graph=True, create_graph=True)[0]
            for i in range(q.size(1))
        ], dim=1)

        C = Hv_field @ q_dot.unsqueeze(2) - KE_field.unsqueeze(2)

        return C
    
    def _grad_potential(self, V, q):
        g = grad(V.sum(), q, create_graph=True)[0]
        return g
        
        
    def _Euler_Lagrange(self, H, c, g, tau):
        n,d = tau.size()
        q_ddot = torch.solve(tau.view(n,d,1) -c.view(n,d,1) - g.view(n,d,1), H)[0] 
        
        return q_ddot.squeeze(2)
   
    def _loss(self, output, target):
        criterion = nn.MSELoss()
        return criterion(output, target)

class LODE_T(nn.Module, Euler_Lagrange_Module):
    """
    Acrobot and Double-Pendulum implementations    
    Manifold here is a torus T² = S¹xS¹ (for n_angles = 2) and consequently the number of 
    input paramters of the inertia matrix will be twice the number of angles: 
        theta1, theta2 ---> (cos(theta1), sin(theta1)), (cos(theta2), sin(theta2))
        
    Data structure: tensor[mxn] where m is the bacth size and n the number of elements of the state-space
    """
    def __init__(self, n_angles, n_layers, n_neurons, activation=None, embedding=True, ln=False):
        super().__init__()
        H_params = int(n_angles*2)
        output_size = None
        self.embedding = embedding
        FF = MLP(H_params, output_size, n_layers, n_neurons, activation, ln)    
        
        RB = Rigid_Body(FF, n_angles, n_neurons)
        self.Rigid_Body = RB
    
    def angle_embedding(self, q):
        x = torch.cos(q)
        y = torch.sin(q)
        new_q = torch.cat((x,y), 1)
        return new_q  
  
    def forward(self, t, x):
        q, q_dot, tau = self._get_data(x)
        with torch.set_grad_enabled(True):
            q = q.requires_grad_(True)
            if self.embedding:
                new_q = self.angle_embedding(q)
            else:
                new_q = q
            #Get Inertia Matrix
            H, V = self.Rigid_Body(new_q)
            #Get coriolis
            c = self._coriolis(H, q, q_dot)
            g = self._grad_potential(V, q)
        
        ddx_dt = self._Euler_Lagrange(H, c, g, tau)
        dx_dt = q_dot
        du_dt = torch.zeros_like(q_dot)
        dx_dt = torch.cat((dx_dt, ddx_dt, du_dt), dim=1)
        return dx_dt
    
    def kinetic_energy(self, H, q_dot):
        q_dot = q_dot.unsqueeze(1)
        T = 0.5*(q_dot @ H @ torch.transpose(q_dot, 1, 2))
        return torch.squeeze(T, 2)

    def get_matrices(self, x):
        d = x.size(1)//2
        q, q_dot = torch.split(x, [d,d], dim = 1)
        new_q = self.angle_embedding(q)
        H, V = self.Rigid_Body(new_q)
        T = self.kinetic_energy(H, q_dot)
        return H.detach(), V.detach(), T.detach()
    
    def momentum(self, x):
        d = x.size(1)//2
        q, q_dot = torch.split(x, [d,d], dim = 1)
        new_q = self.angle_embedding(q)
        H, _ = self.Rigid_Body(new_q)
        p = H @ q_dot.unsqueeze(1)
        return torch.cat((q, p.squeeze(1)), 1).detach()

    def loss(self, output, target):
        return self._loss(output, target)
    
class LODE_RT(nn.Module, Euler_Lagrange_Module):
    """
    Cartpole implementation - Rotational and translational coordinates    
    Manifold here hybrid R^m x T^n (for n_angles = 1) and consequently the number of 
    input paramters of the inertia matrix will be twice the number of angles: 
        r, theta ---> r, (cos(theta), sin(theta))
        
    Data structure: tensor[mxn] where m is the bacth size and n the number of elements of the state-space
    """
    def __init__(self, m, n, n_layers, n_neurons, activation=None, embedding=True, ln=False):
        super().__init__()
        H_params = m+int(2*n)
        self.d_f = m + n
        output_size = None
        self.embedding = embedding
        self.m = m
        self.n = n
        FF = MLP(H_params, output_size, n_layers, n_neurons, activation, ln)    

        RB = Rigid_Body(FF, self.d_f, n_neurons)
        self.Rigid_Body = RB
    
    def angle_embedding(self, q):
        r = q[:,:self.m]
        theta = q[:,self.m:] 
        x = torch.cos(theta)
        y = torch.sin(theta)
        new_q = torch.cat((r,x,y), 1)
        return new_q  
  
    def forward(self, t, x):
        q, q_dot, tau = self._get_data(x)
        with torch.set_grad_enabled(True):
            q = q.requires_grad_(True)
            if self.embedding:
                new_q = self.angle_embedding(q)
            else:
                new_q = q
            #Get Inertia Matrix
            H, V = self.Rigid_Body(new_q)
            #Get coriolis
            c = self._coriolis(H, q, q_dot)
            g = self._grad_potential(V, q)
        
        ddx_dt = self._Euler_Lagrange(H, c, g, tau)
        dx_dt = q_dot
        du_dt = torch.zeros_like(q_dot)
        dx_dt = torch.cat((dx_dt, ddx_dt, du_dt), dim=1)
        return dx_dt
    
    def kinetic_energy(self, H, q_dot):
        q_dot = q_dot.unsqueeze(1)
        T = 0.5*(q_dot @ H @ torch.transpose(q_dot, 1, 2))
        return torch.squeeze(T, 2)

    def get_matrices(self, x):
        d = x.size(1)//2
        q, q_dot = torch.split(x, [d,d], dim = 1)
        new_q = self.angle_embedding(q)
        H, V = self.Rigid_Body(new_q)
        T = self.kinetic_energy(H, q_dot)
        return H.detach(), V.detach(), T.detach()
    
    def loss(self, output, target):
        return self._loss(output, target)  

class ODE(nn.Module):
    def __init__(self, model, odeint, h):
        super().__init__()
        self.h = h
        self.model = model
        self.odeint = odeint
        
    def forward(self, x):
        dev = x.device
        d_f = len(x[0] + 1)//3
        t_eval = torch.tensor([0, self.h]).to(dev)
        g = lambda x: odeint(self.model, x, t_eval, method=self.odeint)
        return torch.squeeze(g(x)[1,:, :-d_f])