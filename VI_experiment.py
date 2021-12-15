#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 04:15:49 2021

@author: saul
"""

# %%
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

from Code.models import cartpole, acrobot, pendulum
import matplotlib.pyplot as plt
from Code.VI import VI_trajectory
import torch
import numpy as np
from Code.data import generate_trajectory, data
# %% Acrobot
t=500
a = 0.5 #midpoint rule
h1 = 0.01 #time step
N1 = int(t/h1)
N_1 = int((N1/t))
x0=[np.pi/2 ,np.pi/2, 0, 0]
u=torch.zeros(size=(N1,1))
tol = 1e-10
# VI trajectory:
xc_vi_1, coords=VI_trajectory("acrobot", tol, x0, N1, h1, a).trajectory(u) 
# RK4 trajectory:
xc_rk4_1, x_next =generate_trajectory("acrobot", x0, N1, h1).get_trajectory()


Energyc_rk4_1 = acrobot().energy(x_next[::N_1])
Energyc_vi_1 = acrobot().energy(coords[::N_1])

True_Energy_acro = acrobot().energy(np.array(x0))
       
h2 = 0.1
N2 = int(t/h2)
N_2 = int((N2/t))
u=torch.zeros((N2,1))
# VI trajectory:
xc_vi_2, coords=VI_trajectory("acrobot", tol, x0, N2, h2, a).trajectory(u) 
# RK4 trajectory:
xc_rk4_2, x_next =generate_trajectory("acrobot", x0, N2, h2).get_trajectory()


Energyc_rk4_2 = acrobot().energy(x_next[::N_2])
Energyc_vi_2 = acrobot().energy(coords[::N_2])


h3 = 0.05
N3 = int(t/h3)
N_3 = int((N3/t))
u=torch.zeros((N3,1))
# VI trajectory:
xc_vi_3, coords=VI_trajectory("acrobot", tol, x0, N3, h3, a).trajectory(u) 
# RK4 trajectory:
xc_rk4_3, x_next =generate_trajectory("acrobot", x0, N3, h3).get_trajectory()


Energyc_rk4_3= acrobot().energy(x_next[::N_3])
Energyc_vi_3 = acrobot().energy(coords[::N_3])

E_acro_rk4 = [Energyc_rk4_1, Energyc_rk4_2, Energyc_rk4_3]
E_acro_vi = [Energyc_vi_1, Energyc_vi_2, Energyc_vi_3]


# %% CartPole - Forced non conservative system
t=500
a = 0.5 #midpoint rule
h1 = 0.01 #time step
N1 = int(t/h1)
N_1 = int((N1/t))
x0=[2 , np.pi/2 , 3, 0]
u=torch.zeros(size=(N1,1))
tol = 1e-7
True_Energy_cart = cartpole().energy(np.array(x0))
# VI trajectory:
xc_vi_1, coords=VI_trajectory("cartpole", tol, x0, N1, h1, a).trajectory(u) 
# RK4 trajectory:
xc_rk4_1, x_next = generate_trajectory("cartpole", x0, N1, h1).get_trajectory()


Energyc_rk4_1 = cartpole().energy(x_next[::N_1])
Energyc_vi_1 = cartpole().energy(coords[::N_1])


       
h2 = 0.1
N2 = int(t/h2)
N_2 = int((N2/t))
u=torch.zeros((N2,1))
# VI trajectory:
xc_vi_2, coords=VI_trajectory("cartpole", tol, x0, N2, h2, a).trajectory(u) 
# RK4 trajectory:
xc_rk4_2, x_next =generate_trajectory("cartpole", x0, N2, h2).get_trajectory()


Energyc_rk4_2 = cartpole().energy(x_next[::N_2])
Energyc_vi_2 = cartpole().energy(coords[::N_2])


h3 = 0.05
N3 = int(t/h3)
N_3 = int((N3/t))
u=torch.zeros((N3,1))
# VI trajectory:
xc_vi_3, coords=VI_trajectory("cartpole", tol, x0, N3, h3, a).trajectory(u) 
# RK4 trajectory:
xc_rk4_3, x_next =generate_trajectory("cartpole", x0, N3, h3).get_trajectory()


Energyc_rk4_3= cartpole().energy(x_next[::N_3])
Energyc_vi_3 = cartpole().energy(coords[::N_3])

E_cart_rk4 = [Energyc_rk4_1, Energyc_rk4_2, Energyc_rk4_3]
E_cart_vi = [Energyc_vi_1, Energyc_vi_2, Energyc_vi_3]


# %% Plots

fig, axs = plt.subplots(1,2, figsize=(17.5, 7), dpi = 600)
N_1 = int((N1/t))
N_2 = int((N2/t))
N_3 = int((N3/t))
axs[0].plot(E_acro_rk4[0],'--', label='Runge-kutta 4, h = 0.01',color='y')
axs[0].plot(E_acro_rk4[1], '--', label='Runge-kutta 4, h = 0.1' ,color='b')
axs[0].plot(E_acro_rk4[2], '--', label='Runge-kutta 4, h = 0.05' ,color='r')
axs[0].plot(E_acro_vi[0],label='Variational Midpoint, h = 0.01', color='y')
axs[0].plot(E_acro_vi[1], label='Variational Midpoint, h = 0.1', color='b')
axs[0].plot(E_acro_vi[2], label='Variational Midpoint, h = 0.05', color='r')
axs[0].legend()
axs[0].set_title(r'Free Acrobot')
axs[0].set_xlabel('Time [s]')
axs[0].set_ylabel('Energy')
axs[0].set_ylim([2, 7]) 

axs[1].plot(E_cart_rk4[0],'--', label='Runge-kutta 4, h = 0.01',color='y')
axs[1].plot(E_cart_rk4[1], '--', label='Runge-kutta 4, h = 0.1' ,color='b')
axs[1].plot(E_cart_rk4[2], '--', label='Runge-kutta 4, h = 0.05' ,color='r')
axs[1].plot(E_cart_vi[0],label='Variational Midpoint, h = 0.01', color='y')
axs[1].plot(E_cart_vi[1], label='Variational Midpoint, h = 0.1', color='b')
axs[1].plot(E_cart_vi[2], label='Variational Midpoint, h = 0.05', color='r')
axs[1].legend()
axs[1].set_ylim([2, 6]) 
axs[1].set_xlabel('Time [s]')
axs[1].set_title(r'Free Cartpole')
axs[1].set_ylabel('Energy')

plt.savefig('figures/energy_dissipation_VIvsrk4'.format('png'))

# %%
