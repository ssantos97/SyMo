import torch
from torch import nn
import copy

from Code.broyden_layer import Broyden_RootFind, BroydenLayer
from Code.newton_layer import Newton_RootFind, NewtonLayer
from Code.NN import Euler_Lagrange_Module

class rootfind(nn.Module):
    def __init__(self, model, method, forward_tol, forward_maxiter, analyse = False):
        super().__init__()
        self.model = model 
        self.forward_tol = forward_tol
        self.forward_maxiter = forward_maxiter
        self.method = method
        self.analyse = analyse
        if method == "Broyden":
            self.RootFind = Broyden_RootFind
        elif method == 'Newton':
            self.RootFind = Newton_RootFind
        else:
            raise NameError('Only Newton or Broyden are available. Cannot recognize:' + str(method))

    def initial_guess(self, *args):
        """
        Explicit Euler integration where the velocity is approximated by q_dot = (q - q_past)/h
        q_next0 = q + h*q_dot = 2q - q_past 
        """
        q_past, q, u_past, u, u_next = args
        q_next0 = 2*q - q_past
        return q_next0

    def forward(self, x):
        d = x.shape[1] // 5 #number of degrees of freedom
        q_past, q, u_past, u, u_next = torch.split(x, d, 1)
        data = [q_past, q, u_past, u, u_next]
        
        with torch.no_grad():
            q_next0 = self.initial_guess(*data)

        if self.training:    
            func_copy = copy.deepcopy(self.model)
            for params in func_copy.parameters():
                params.requires_grad_(False)  
        else:
            func_copy = self.model
        args = [q_past, q, u_past, u, u_next, self.forward_tol, self.forward_maxiter]
        
        if self.training:
            if self.method == 'Newton':
                q_next, nstep, diff = self.RootFind.apply(func_copy, q_next0, *args)
                #Adding DEL to the computation graph
                new_q_next = self.RootFind.fval(self.model, q_next, *data) + q_next #This is a no-op in terms of q_next, since the value of DEL for the root should be 0
                #send q_next to get gradient and not output
                new_q_next = NewtonLayer.apply(func_copy, new_q_next, *data)
            
            elif self.method == 'Broyden':
                q_next, nstep, diff, C, DT = self.RootFind.apply(func_copy, q_next0, *args)
                #Adding DEL to the computation graph
                new_q_next = self.RootFind.fval(self.model, q_next, *data) + q_next #This is a no-op in terms of q_next, since the value of DEL for the root should be 0

                B = get_est_jac(C, DT)      #get inverse jacobian approximation 
                args = [q_past, q, u_past, u, u_next]
                new_q_next = BroydenLayer.apply(func_copy, new_q_next, B, *args)
        
        if self.training:
            return new_q_next, nstep, diff #for analysis
        else:
            return self.RootFind.apply(func_copy, q_next0, *args)[0]

def get_est_jac(C, DT):
    #Makes the update for the inverse Jacobian
    dev = C.device
    bsz, d_f, _ = C.size()
    x = torch.eye(d_f)
    x = x.reshape((1, d_f, d_f))
    I = x.repeat(bsz, 1, 1).to(dev)
    return -I + C@DT