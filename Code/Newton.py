import torch
import operator

def Newton(func, x0, jac, tol, maxiter, *args):
    """
    x0 = (bsz, d_f) where bsz is batch size and d_f the number of degrees of freedom
    func  = (bsz, d_f)
    jac = (bsz, d_f, d_f) 
    """
    if tol <= 0:
        raise ValueError("tol too small (%g <= 0)" % tol)
    maxiter = operator.index(maxiter)
    if maxiter < 1:
        raise ValueError("maxiter must be greater than 0")
    bsz, d_f = x0.size() 
    p = x0.view(bsz, d_f, 1).clone().detach()
    iteration = 0
    protect_thres = 1e9 * bsz
    lowest = p
    # first evaluate fval
    fval = func(p.view(bsz, d_f))
    new_objective = torch.norm(fval).item()
    if new_objective == 0:
        #all zeros were found
        return {"result": lowest.view(bsz, d_f),
        "nstep": iteration,
        "diff": new_objective,
        "diff_detail": torch.norm(fval, dim=1)}    
    
    lowest_objective = new_objective
    lowest_gx = fval
    lowest_step =iteration
    # Newton-Raphson method
    while iteration < maxiter: 
        fder = jac(p.view(bsz, d_f))
        # Newton step
        dp = torch.inverse(fder) @ fval.view(bsz, d_f, 1)
        
        if torch.norm(dp) > protect_thres:
            msg = "Singular Jacobian"
            raise RuntimeError(msg)
        
        p -= dp
        iteration += 1
        fval = func(p.view(bsz, d_f))
        new_objective = torch.norm(fval).item()
    
        if new_objective < lowest_objective:
            lowest_gx = fval
            lowest_objective = new_objective
            lowest = p
            lowest_step =iteration
        if new_objective < tol:
            break
    return {"result": lowest.view(bsz, d_f),
        "nstep": iteration,
        "lowest_step" : lowest_step,
        "diff": lowest_objective,
        "diff_detail": torch.norm(lowest_gx, dim=1)}    