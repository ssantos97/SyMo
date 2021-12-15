from Code.root_find import rootfind
from Code.NN import ODE
import torch
import numpy as np

def rungekutta2(f, y0, t, dev, args=()):
    n = len(t)
    y = torch.zeros((n, len(y0))).to(dev)
    y[0] = y0
    if args is None:
        args = torch.zeros((n, int(len(y0)/2))).to(dev)
    else: 
        args = torch.tensor(args).float().to(dev)
    for i in range(n - 1):
        h = t[i+1] - t[i]
        x1 = torch.cat((y[i], args[i]),0)[None,:]
        k1 = torch.cat((y[i] + (h/2)*f(t[i], x1)[0, :len(y0)].detach(), args[i]),0)[None, :]
        k2 = f(t[i] + (h/2), k1)[0, :len(y0)].detach()

        y[i+1] = y[i] + h*k2
    
    return y[1:]
    
def rungekutta4(f, y0, t, dev, args=()):
    n = len(t)
    y = torch.zeros((n, len(y0))).to(dev)
    y[0] = y0
    if args is None:
        args = torch.zeros((n, int(len(y0)/2))).to(dev)
    else:
        args = torch.tensor(args).float().to(dev)
    for i in range(n - 1):
        h = t[i+1] - t[i]
        x1 = torch.cat((y[i], args[i]),0)[None,:]
        k1 = f(t[i], x1)[0, :len(y0)].detach()
        x2 = torch.cat((k1, args[i]),0)[None,:]
        k2 = f(t[i] + h / 2., x1 + x2 * h / 2.)[0, :len(y0)].detach()
        x3 = torch.cat((k2, args[i]),0)[None,:]
        k3 = f( t[i] + h / 2., x1 + x3 * h / 2.)[0, :len(y0)].detach()
        x4 = torch.cat((k3, args[i]),0)[None,:]
        k4 = f(t[i] + h, x1 + x4 * h)[0, :len(y0)].detach()
        y[i+1] = y[i] + (h / 6.) * (k1 + 2*k2 + 2*k3 + k4)
    return y[1:]
 


def integrate_ODE(nn, ode_solver, x0, n_int, time_step, device, us=None):
    t_span = [0, n_int*time_step]
    t_eval = np.linspace(t_span[0], t_span[1] , n_int + 1)
    
    if ode_solver == "midpoint":
        y_pred = rungekutta2(nn.eval(), x0, t_eval, device, us)
    elif ode_solver == "rk4":
        y_pred = rungekutta4(nn.eval(), x0, t_eval, device, us)
    
    return y_pred

def implicit_integration_DEL(root_find, n_int, h, nn, x0, pred_tol, pred_maxiter, device, us = None):
    """
    Takes as an input 1 initial condition, 1 sequence of controls (optional) and all the other obvious arguments
    """
    x0 = x0.squeeze(0).to(device)
    d_f = int(len(x0)/2)
    model = rootfind(nn, root_find, pred_tol, pred_maxiter).to(device).eval()
    if us is not None:
        assert len(us) == n_int +2, "The controls used for integration need n_time_steps + 2 samples. Note that (u(t-1), u(t), u(t+1)) is required for one step integration!"
        us = torch.tensor(us)
        us = us.to(device).float().to(device)
    else: 
        us = torch.zeros(n_int + 2, d_f).to(device)
    q_past, q = torch.split(x0, d_f,0)
    pred = []
    for step in range(n_int):
        u_past = us[step]
        u = us[step +1]
        u_next = us[step + 2]
        x = torch.cat((q_past, q, u_past, u, u_next), 0)[None, :] 
        q_next = model(x).detach()[0]
        #reconstruct pose
        v_next = (q_next - q)/h
        state = torch.cat((q_next, v_next), 0)
        pred.append(state)
        q_past = q
        q = q_next
    y_pred = torch.stack(pred) 
    return y_pred