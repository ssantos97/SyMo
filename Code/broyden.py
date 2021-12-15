import torch
import numpy as np

def line_search(update, x0, g0, g):
    """
    Line search with s=1. Could have been used armijo scalar search (scipy)
    """

    s = 1.
    x_est = x0 + s * update
    g0_new = g(x_est).clone().detach()
    return x_est - x0, g0_new - g0

def rmatvec(part_C, part_DT, x):
    # Compute x^T(-I + CD^T)
    # x: (bsz, d_f)
    # part_Us: (N, d_f, maxiter)
    # part_VTs: (N, maxiter, d_f)
    if part_C.nelement() == 0:
        return -x
    xTU = torch.einsum('bi, bid -> bd', x, part_C)   # (bsz, maxiter)
    return -x + torch.einsum('bd, bdi -> bi', xTU, part_DT)    # (bsz, d_f)


def matvec(part_C, part_DT, x):
    # Compute (-I + CD^T)x
    # x: (bsz, d_f)
    # part_C: (bsz, d_f, maxiter)
    # part_DT: (bsz, maxiter, d_f)
    if part_C.nelement() == 0:
        return -x
    VTx = torch.einsum('bdi, bi -> bd', part_DT, x)  # (bsz, maxiter)
    return -x + torch.einsum('bid, bd -> bi', part_C, VTx)     # (bsz, d_f)

def perform_update(C, DT, nz_dgx, nstep, delta_x, delta_gx):
    part_C, part_DT = C[nz_dgx, :, :nstep-1], DT[nz_dgx, :nstep-1]
    vT = rmatvec(part_C, part_DT, delta_x[nz_dgx])
    u = (delta_x[nz_dgx] - matvec(part_C, part_DT, delta_gx[nz_dgx])) / torch.einsum('bi, bi -> b', vT, delta_gx[nz_dgx])[:, None]
    vT[vT != vT] = 0
    u[u != u] = 0
    DT[nz_dgx,nstep-1] = vT
    C[nz_dgx,:,nstep-1] = u
    return C, DT




def broyden(g, x0, tol, maxiter):
    # When doing low-rank updates at a (sub)sequence level, we still only store the low-rank updates, 
    # instead of the huge matrices
    bsz, d_f = x0.size()
    
    x_est = x0       # (bsz, d_f)
    device = x_est.device
    gx = g(x_est).clone().detach()   # (bsz, d_f)
    #nz_dgx is used when the difference between 2 iterations is so small that gx is the same leading to delta_gx=0 and hence nans.
    # if gx=0 then all zeros were found
    nz_dgx = (torch.norm(gx, dim=1) != 0)
    nstep = 0

    # For fast calculation of jacobian (approximately) - Sherman-Morison Formula
    C = torch.zeros(bsz, d_f, maxiter +1).to(device) #maxiter + 1 because we need 
    DT = torch.zeros(bsz, maxiter+1, d_f).to(device)
    update = -matvec(C[:,:,:nstep], DT[:,:nstep], gx)
    new_objective = init_objective = torch.norm(gx).item()
    trace = [init_objective]
    # To be used in protective breaks
    protect_thres = 1e9 * bsz
    lowest = new_objective
    lowest_xest, lowest_gx, lowest_step = x_est.clone(), gx.clone(), nstep
    
    while nstep < maxiter:
        delta_x, delta_gx = line_search(update, x_est, gx, g)
        nz_dgx = torch.logical_and(torch.norm(delta_gx, dim=1) != 0, torch.norm(delta_x, dim=1) != 0)
        if not nz_dgx.any(): #all zeros were found
            break
        x_est[nz_dgx] += delta_x[nz_dgx]
        gx[nz_dgx] += delta_gx[nz_dgx]
        nstep += 1
        new_objective = torch.norm(gx).item()
        trace.append(new_objective)
        
        if new_objective < lowest:
            lowest_xest, lowest_gx = x_est.clone().detach(), gx.clone().detach()
            lowest_nz_dgx = nz_dgx
            lowest_delta_x = delta_x
            lowest_delta_gx = delta_gx
            lowest = new_objective
            lowest_step = nstep
        if new_objective < tol:
            break
        if new_objective > init_objective * protect_thres or np.isnan(new_objective):
            break
        
        C, DT = perform_update(C, DT, nz_dgx, nstep, delta_x, delta_gx)
        update[nz_dgx] = -matvec(C[nz_dgx,:,:nstep], DT[nz_dgx,:nstep], gx[nz_dgx]) 
    

    results = {"result": lowest_xest,
    "nstep": nstep,
    "lowest_step": lowest_step,
    "diff":lowest,
    "diff_detail": torch.norm(lowest_gx, dim=1),
    }
    if g.training:
        results['C'], results['DT'] = perform_update(C, DT, lowest_nz_dgx, lowest_step, lowest_delta_x, lowest_delta_gx)
    return results