import torch
from torch.autograd import grad
from torch import nn
from abc import ABC, abstractmethod

class DEL_Module(object):   
    @abstractmethod
    def lagrangian(self, q, qd):
        "Returns the Lagrangian for given position and velocity"
        
    
    def midpoint_rule(self, q, q_next, h):
        a = 0.5 #midpoint
        q_ap = a*q + a*q_next
        return q_ap, (q_next - q)/h
    
    def _kinetic_energy(self, H, q_dot):
        q_dot = q_dot.unsqueeze(1)
        T = 0.5*(q_dot @ H @ torch.transpose(q_dot, 1, 2))
        return torch.squeeze(T, 2)
    
    def _DEL(self, q_next, *args):
        q_past, qk, u_past, u, u_next = args[:-1]
        h = args[-1]
        with torch.set_grad_enabled(True):
            q = qk.requires_grad_(True)
            #Discretize
            q1, qd1 = self.midpoint_rule(q_past, q, h)
            q2, qd2 = self.midpoint_rule(q, q_next, h)
            
            L_past = self.lagrangian(q1, qd1)
            L = self.lagrangian(q2, qd2)
            
            f = grad(L_past.sum() + L.sum(), q, create_graph = True)[0] 

            #midpoint force approximation
            f_right = ((u_past + u)*0.5)*0.5
            f_left = ((u + u_next)*0.5)*0.5
            DEL = f + (f_right + f_left)*h
        return DEL
    
    def _Jacobian(self, q_next, *args):
        """
        Used for the Newton layer and to make prediction in the non end to end SyMo Model
        """
        with torch.set_grad_enabled(True):
            q_next = q_next.requires_grad_(True)
            DEL = self._DEL(q_next, *args)
            DEL.requires_grad_(True)
            bsz, d_f = q_next.size()
            J = []
            for i in range(d_f):
                D1_i = DEL[:, i][:, None]
                J_i = grad(D1_i[:].sum(), q_next, create_graph=True)[0]
                J.append(J_i)
    
            if d_f > 1: #not 1 degree of freedom   
                J_i = torch.stack((J[0], J[1]), 0)
                J = J_i.permute(1, 0, 2)
                J = torch.squeeze(J, 0)
            else: #1 degree_freedom
                J = J_i
            return J.view(bsz, d_f, d_f).detach()

    def _momentums(self, q_past, qk, h):
        q = qk.requires_grad_(True)
        q1, qd1 = self.midpoint_rule(q_past, qk, h)
        L1 = self.lagrangian(q1, qd1)
        p = grad(L1.sum(), q, create_graph = False)[0]
        return torch.cat((q, p), 1).detach()
