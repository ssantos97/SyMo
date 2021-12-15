#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 20:53:18 2021

@author: saul
"""

import importlib
import scipy.integrate
solve_ivp = scipy.integrate.solve_ivp
import numpy as np
import torch
import torch.distributions as tdist
def rungeKutta(dydx, x0, u, h):
    # Iterate for number of iterations

    "Apply Runge Kutta Formulas to find next value of y"
    k1 = h * dydx(None, x0, u)
    k2 = h * dydx(None, x0 + 0.5 * k1, u)
    k3 = h * dydx(None, x0 + 0.5 * k2, u)
    k4 = h * dydx(None, x0 + k3, u)

    # Update next value of y
    y = x0 + (1.0 / 6.0)*(k1 + 2 * k2 + 2 * k3 + k4)
    return y

class odeint(object):
    def __init__(self, model, state, h, u = None):
        self.state = state
        self.u = u
        self.model = model
        self.h = h
        
    def integrate(self):
        # ivp = solve_ivp(fun=lambda t, y:self.model.Lagrangian_analytical(t, y, self.u), t_span=[0, self.h], y0=self.state, method='RK45')
        # state = ivp.y[:, -1]

        dydx = self.model.Lagrangian_analytical
        state = rungeKutta(dydx, self.state, self.u, self.h)
        return state
        
class generate_trajectory(object):
    def __init__(self, model, initial_condition, N, h, u = None):
        self.h = h
        self.initial_condition=np.array(initial_condition)
        self.N=N
        self.n_dim=len(initial_condition)
        if u is None:
            self.u = np.zeros(shape=(N,1))
        else:
            self.u=np.array(u)
        class_ = getattr(importlib.import_module("Code.models"), model)
        self.model = class_
    
    def get_trajectory(self):
        states=np.empty(shape=(self.N, self.n_dim))
        next_states=np.empty(shape=(self.N, self.n_dim))
        states[0] = self.initial_condition
        model=self.model()    
        for i in range(self.N):
            if i != 0:
                states[i,:]=next_states[i-1]
            x_next=odeint(model, states[i], self.h, self.u[i]).integrate()
            next_states[i]=x_next
        return states, next_states
    
    
    
#Confusing code: Make it more interpretable!!!!!!    
class data(object):
    '''
    Creates a discrete trajectory based on an random uniform initial condition and the number of points for each trajectory. The trajectories are 
    calculated via Runge kutta 4 discretization
    
    ---> lower and up are lists
    '''
    def __init__(self, model, upper, lower, h, N, N_test, n, seed, u, noise_std=None, noise_seed = None):
        self.upper = upper # upper bound of the dataset
        self.lower = lower # lower bound of the dataset
        self.h = h # time_step
        self.N_train = N
        self.N_test = N_test
        self.N = N + N_test #number of trajectories
                                     
        self.model = model
        self.n = n +2 # length of each trajectory
        self.u = u #control inputs for an individual trajectory, a scaling random factor is multiplied.
        self.d = (len(upper)//2)*2
        self.u_lim = [lower[-1], upper[-1]]
        self.noise = noise_std
        self.noise_seed = noise_seed
        self.seed = seed
 
    def random_initial_condition(self):
        
        """
        Create a set of random initial conditions and according with the input signal multiplies it to random amplitudes
        
        """
        torch.manual_seed(self.seed)
        #create random amplitudes for input signal - Uniform distribution
        if self.u == "random":
            torch.manual_seed(self.seed)
            u=(self.u_lim[0] - self.u_lim[1]) * torch.rand(self.N,self.n,1) + self.u_lim[1]
        elif self.u == "constant":
            torch.manual_seed(self.seed)
            u_factor = (self.u_lim[0] - self.u_lim[1]) * torch.rand(self.N,1,1) + self.u_lim[1]
            u = torch.ones(size=(self.N, self.n, 1)) *u_factor
        elif self.u == "zeros":
            u = torch.zeros(size=(self.N, self.n, 1))
        torch.manual_seed(self.seed)
        #sample N random initial conditions
        x0 = torch.distributions.Uniform(torch.tensor(self.lower[:self.d]).float(), torch.tensor(self.upper[:self.d]).float()).sample(sample_shape=(self.N, 1))
        x0 = torch.squeeze(x0, 1)

        return x0, u

    def get_continuous_data(self):
        x0, u = self.random_initial_condition()
        traj = torch.empty(size=(self.N, self.n, self.d))
        #continuous data 
        states = []
        states_true=[]
        for i in range(self.N):
            x, next_states = torch.tensor(generate_trajectory(self.model, x0[i], self.n, self.h, u[i]).get_trajectory()) 
            traj[i] = x

        y_train = traj[:, 2:]
        if self.noise is not None:
            sigma = torch.tensor([self.noise]).float()
            mu = torch.tensor([0]).float()
            torch.manual_seed(self.noise_seed)
            n = tdist.Normal(mu, sigma)
            noise = torch.squeeze((n.sample((traj.shape))),-1)
            traj = traj + noise
        return traj, y_train, u.float()
        
    
    def get_DEL_data(self):
        data = self.get_continuous_data()
        data, y, u, = data
        x_train = []
        us = []
        d_f = int(self.d/2)

        if self.model == 'cartpole':
            u = u.view(self.N, self.n,1)
            zero_controls = torch.zeros_like(u)
            x = torch.cat((data[:,:-2, :d_f], data[:, 1:-1, :d_f], u[:, :-2], zero_controls[:, :-2], u[:, 1:-1], zero_controls[:, 1:-1],u[:, 2:],  zero_controls[:, 2:]),2)
        elif self.model == 'acrobot':
            u = u.view(self.N, self.n,1)
            zero_controls = torch.zeros_like(u)
            x = torch.cat((data[:,:-2, :d_f], data[:, 1:-1, :d_f], zero_controls[:, :-2], u[:, :-2], zero_controls[:, 1:-1], u[:, 1:-1], zero_controls[:, 2:],u[:, 2:]),2)
        elif self.model == 'pendulum':
            u = u.view(self.N, self.n,1)
            x = torch.cat((data[:,:-2, :d_f], data[:, 1:-1, :d_f], u[:, :-2], u[:, 1:-1], u[:, 2:]), 2)
        
        print('Dataset generated') 
        x_train, x_test = x[:self.N_train], x[self.N_train:]  
        y_train, y_test = y[:self.N_train], y[self.N_train:]
        u_train, u_test = u[:self.N_train], u[self.N_train:]     
        if self.N_test == 0:
            return x_train, y_train, u_train
        else:
            return x_train, y_train, u_train, x_test, y_test, u_test


    def get_NODE_data(self):
        us = []
        d_f = int(self.d/2)
        data, y, u  = self.get_continuous_data()
        x = data[:, 1:-1]
        u = u[:, 1:-1]

        x_train, x_test = x[:self.N_train], x[self.N_train:]  
        y_train, y_test = y[:self.N_train], y[self.N_train:]
        u_train, u_test = u[:self.N_train], u[self.N_train:]  
        if self.N_test == 0:
            return x_train, y_train, u_train
        else:
            return x_train, y_train, u_train, x_test, y_test, u_test
    
    # def get_DHP_data(self):
    #     data = self.get_continuous_data()
    #     data, y, u, = data
    #     x_train = []
    #     us = []
    #     d_f = int(self.d/2)

        
    #     for i in range(self.N):
    #             if self.model == 'cartpole':
    #                 x = torch.cat((data[i,:-2, :d_f], data[i, 1:-1, d_f:], data[i, 1:-1, :d_f], u[i, :-2], torch.tensor([0.]), u[i, 1:-1], torch.tensor([0.]), u[i, 2:], torch.tensor([0.]) ),0)
    #             elif self.model == 'acrobot':
    #                 x = torch.cat((data[i-2,:d_f], data[i-1, d_f:], data[i-1, :d_f], torch.tensor([0.]), torch.tensor([u[i-2]]).float(), torch.tensor([0.]), torch.tensor([u[i-1]]).float(), torch.tensor([0.]), torch.tensor([u[i]]).float(), ),0)
    #             elif self.model == 'pendulum':
    #                 u = u.view(self.N, self.n,1)
    #                 x = torch.cat((data[i,:-2, :d_f], data[i, 1:-1, d_f:], data[i, 1:-1, :d_f], u[i, :-2], u[i, 1:-1], u[i, 2:]),1)
    #             x_train.append(x)
    #     print('Dataset generated') 
    #     x = torch.stack(x_train)
    #     x_train, x_test = x[:self.N_train], x[self.N_train:]  
    #     y_train, y_test = y[:self.N_train], y[self.N_train:]
    #     u_train, u_test = u[:self.N_train], u[self.N_train:]     
    #     if self.N_test == 0:
    #         return x_train.reshape(int(self.N_train*(self.n-2)), int(6*d_f)), y_train.reshape(int(self.N_train*(self.n-2)), int(2*d_f)), u_train.view(int(self.n*self.N_train),1) 
    #     else:
    #         return x_train.reshape(int(self.N_train*(self.n-2)), int(6*d_f)), y_train.reshape(int(self.N_train*(self.n-2)), int(2*d_f)), u_train.view(int(self.n*self.N_train),1), x_test.reshape(int(self.N_test*(self.n-2)), int(6*d_f)), y_test.reshape(int(self.N_test*(self.n-2)), int(2*d_f)), u_test.view(int(self.n*self.N_test),1)

