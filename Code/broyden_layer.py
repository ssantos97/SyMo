import numpy as np
import torch
from torch import nn
from torch.autograd import Function
from Code.broyden import broyden


class Broyden_RootFind(Function):
    """ Generic layer module that uses bad broyden's method to find the solution """
    @staticmethod
    def fval(func, q_next, *args):
        q_past, q, u_past, u, u_next = args

        return func(torch.cat((q_past, q, q_next, u_past, u, u_next), 1))

    @staticmethod
    def broyden_find_root(func, q_next0, tol, maxiter, *args):

        g_func = lambda q_next: Broyden_RootFind.fval(func, q_next, *args)

        results = broyden(g_func, q_next0, tol, maxiter)
        return results

    @staticmethod
    def forward(ctx, func, q_next0, *args):
        bsz, d_f = args[0].size()
        root_find = Broyden_RootFind.broyden_find_root
        ctx.args_len = len(args)
        q_past, q, u_past, u, u_next = args[:-2]
        tol = args[-2]*np.sqrt(bsz*d_f)
        maxiter = args[-1]
        with torch.no_grad():
            guess = q_next0.clone().detach()
            args = [q_past, q, u_past, u, u_next] 
            results = root_find(func, guess, tol, maxiter, *args)  
            return results

    @staticmethod
    def backward(ctx, grad_q_next):
        grad_args = [None for _ in range(ctx.args_len)]
        return (None, grad_q_next, *grad_args)
  

class BroydenLayer(Function):
    """    
    Call this line to apply this implicit layer
        self.NewtonLayer.apply(self.func_copy, ...)
        
    """
    @staticmethod
    def forward(ctx, func_copy, q_next, B, *args):
        q_past, q, u_past, u, u_next = args[:-2]
        ctx.args_len = len(args)
        # ctx.tol = tol
        ctx.B = B
        # ctx.maxiter = maxiter
        ctx.save_for_backward(q_next, q_past, q, u_past, u, u_next)
        ctx.func = func_copy
        ctx.args_len = len(args) + 2
        return q_next

    @staticmethod
    def backward(ctx, grad):
        torch.cuda.empty_cache()
        # grad should have dimension (bsz x d_f)
        grad = grad.clone()
        q_next, q_past, q, u_past, u, u_next = ctx.saved_tensors
        bsz, d_f = q_next.size()
        func = ctx.func
        q_next = q_next.clone().detach()
        q_past = q_past.clone().detach()
        q = q.clone().detach()
        u_past = u_past.clone().detach()
        u = u.clone().detach()
        u_next = u_next.clone().detach()
        args = [q_past, q, u_past, u, u_next]
        
        # with torch.enable_grad():
        #     y = Broyden_RootFind.fval(func, q_next, *args)

        # def g(x):
        #     y.backward(x, retain_graph=True)   # Retain for future calls to g
        #     JTx = q_next.grad.clone().detach()
        #     q_next.grad.zero_()
        #     return JTx + grad
        # maxiter = ctx.maxiter

        # tol = ctx.tol*np.sqrt(bsz * d_f)

        #Initial Guess
        # dl_df_est = torch.zeros_like(grad)
        # result_info = broyden(g, dl_df_est, tol, maxiter)
        # dl_df_est = result_info['result']
        # y.backward(torch.zeros_like(dl_df_est), retain_graph=False)
        grad_args = [None for _ in range(ctx.args_len)]
        dl_df_est =  - grad.view(bsz, 1, d_f) @ ctx.B
        return (None, dl_df_est.view(bsz, d_f), *grad_args)