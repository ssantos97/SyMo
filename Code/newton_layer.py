import numpy as np
import torch
from torch.autograd import Function
from Code.Newton import Newton


class Newton_RootFind(Function):
    """ Generic newton layer module that uses Newton's method to find the solution """
    @staticmethod
    def fval(func, q_next, *args):
        q_past, q, u_past, u, u_next = args
        return func(torch.cat((q_past, q, q_next, u_past, u, u_next), 1))

    @staticmethod 
    def jacobian(jac, q_next, *args):
        q_past, q, u_past, u, u_next = args
        return jac(torch.cat((q_past, q, q_next, u_past, u, u_next), 1))

    @staticmethod
    def newton_find_root(func, v_next0, jac, tol, maxiter, *args):

        g_func = lambda q_next: Newton_RootFind.fval(func, q_next, *args)
        jac_func = lambda q_next: Newton_RootFind.jacobian(jac, q_next, *args)

        return Newton(g_func, v_next0, jac_func, tol, maxiter)

    @staticmethod
    def forward(ctx, func, q_next0, *args):
        bsz, d_f = args[0].size()
        root_find = Newton_RootFind.newton_find_root
        ctx.args_len = len(args)
        q_past, q, u_past, u, u_next = args[:-2]
        tol = args[-2]*np.sqrt(bsz*d_f)
        maxiter = args[-1]

        jac = func.jacobian
        with torch.no_grad():
            guess = q_next0.clone().detach()
            args = [q_past, q, u_past, u, u_next] 
            results = root_find(func, guess, jac, tol, maxiter, *args)  
            q_next = results['result'].clone().detach()
            nstep = torch.tensor(results['nstep'])
            diff = torch.tensor(results['diff'])
            return q_next, nstep, diff
    @staticmethod
    def backward(ctx, grad_q_next):
        grad_args = [None for _ in range(ctx.args_len)]
        return (None, grad_q_next, *grad_args)
  

class NewtonLayer(Function):
    """    
        self.NewtonLayer.apply(self.func_copy, ...)
        
    """
    @staticmethod
    def forward(ctx, func_copy, q_next, *args):
        q_past, q, u_past, u, u_next = args
        ctx.save_for_backward(q_next, q_past, q, u_past, u, u_next)
        ctx.func = func_copy
        ctx.jac = func_copy.jacobian
        ctx.args_len = len(args) + 1
        return q_next

    @staticmethod
    def backward(ctx, grad):
        torch.cuda.empty_cache()

        # grad should have dimension (bsz x d_f)
        bsz, d_f = grad.size()
        grad = grad.clone()
        q_next, q_past, q, u_past, u, u_next = ctx.saved_tensors
        jac = ctx.jac
        q_next = q_next.clone().detach()
        q_past = q_past.clone().detach()
        q = q.clone().detach()
        u_past = u_past.clone().detach()
        u = u.clone().detach()
        u_next = u_next.clone().detach()
        args = [q_past, q, u_past, u, u_next]
        
        J = Newton_RootFind.jacobian(jac, q_next, *args)

        dl_df =  - grad.view(bsz, 1, d_f) @ torch.inverse(J)
        
        grad_args = [None for _ in range(ctx.args_len)]
        return (None, dl_df.view(bsz, d_f), *grad_args)