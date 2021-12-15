#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 23:01:38 2021

@author: saul
"""

from abc import ABC, abstractmethod
from scipy.misc import derivative
import numpy as np
from Code.data import generate_trajectory
from numpy.linalg import inv, pinv
import torch
import importlib

class discrete_model(object):
    def __init__(self, model, a, h):
        class_ = getattr(importlib.import_module("Code.models"), model)
        self.model = class_  
        self.h = h
        self.a = a
        
    def Lagrangian_integration(self, q0, q1):
        q0, q1 = np.array(q0), np.array(q1)
        theta = (1-self.a)*q0 + self.a*q1
        dtheta = (q1 - q0)/self.h
        return theta, dtheta
    
    def DEL(self, q_past, q, q_next, u_past, u, u_next):
        self.d = len(q_past)
        q_past1, q_past2 = q_past[0]
        q1, q2 = q[0]
        q_next1, q_next2 = q_next[0]
        
        inp = [q_past1, q_past2, q1, q2, q_next1, q_next2]
        DEL0 = self.partial_derivative(self.func, 2, inp) 
        DEL1 = self.partial_derivative(self.func, 3, inp) 
        
        return np.array([[DEL0, DEL1]])
    
    def partial_derivative(self, func, var=0, point=[]):
        args = point[:]
        def wraps(x):
            args[var] = x
            return func(*args)
        return derivative(wraps, point[var], dx = 1e-6, order = 3)
    
    def func(self, *args):
        q_past1, q_past2 = args[0], args[1]
        q1, q2 = args[2], args[3]
        q_next1, q_next2 = args[4], args[5]
        q_past = np.array([q_past1, q_past2])
        q = np.array([q1, q2])
        q_next = np.array([q_next1, q_next2])
        
        
        q1, qd1 = self.Lagrangian_integration(q_past, q)
        q2, qd2 = self.Lagrangian_integration(q, q_next)
        state1 = np.concatenate((q1, qd1), 0)
        state2 = np.concatenate((q2, qd2), 0)
        
        L1 = self.model().kinetic_energy(state1).item() - self.model().potential_energy(state1).item()
        L2 = self.model().kinetic_energy(state2).item() - self.model().potential_energy(state2).item()
        
        return L1*self.h + L2*self.h
    
    def Jacobian(self, q_past, q, q_next, u_past, u, u_next):
        q_past1, q_past2 = q_past[0]
        q1, q2 = q[0]
        q_next1, q_next2 = q_next[0]
        inp1 = [q_past1, q_past2, q1, q2, q_next1, q_next2, 1]
        inp2 = [q_past1, q_past2, q1, q2, q_next1, q_next2, 2]
        
        J11 = self.partial_derivative(self.Jac_fun, 4, inp1) 
        J12 = self.partial_derivative(self.Jac_fun, 5, inp1) 
        J21 = self.partial_derivative(self.Jac_fun, 4, inp2) 
        J22 = self.partial_derivative(self.Jac_fun, 5, inp2) 
        
        return np.array([[J11, J12], [J21, J22]])
        
    def Jac_fun(self, *args):
        "The Jacobian of the DEL for the root finding algorithm is correspondent with the D2D1L(q, q_next) term"
        q_past1, q_past2 = args[0], args[1]
        q1, q2 = args[2], args[3]
        q_next1, q_next2 = args[4], args[5]
        
        d_f = args[6]
        inp = [q_past1, q_past2, q1, q2, q_next1, q_next2]
        if d_f == 1: 
            DEL = self.partial_derivative(self.func, 2, inp)
        else:
            DEL = self.partial_derivative(self.func, 3, inp) 
        
        return DEL
    
    
class VI_trajectory(object):
    def __init__(self, model, tol, x0, N, h, a):
        self.model = model
        self.tol = tol
        self.x0 = x0
        self.N  = N
        self.h = h
        self.a = a
        self.f = discrete_model(self.model, self.a, self.h)
    def Iteration(self, init, q_past, q, u_past, u, u_next):     
        lastX = init
        error = 9e9
        while (error > self.tol):  # this is how you terminate the loop - note use of abs()
            newY = self.f.DEL(q_past, q, lastX, u_past, u, u_next)
            J = self.f.Jacobian(q_past, q, lastX, u_past, u, u_next)
            newX = lastX.T - inv(J)@ newY.T # update estimate using N-R
            error = np.max(np.abs(lastX.T - newX ))
            lastX = newX.T
        return newX

    def get_root(self, q_past, q, u_past, u, u_next):
        #Initial guess:
        q_past = np.array([q_past], dtype=np.double)
        q = np.array([q])
        dq = (q - q_past)
        x0 = q + dq
        q_next = self.Iteration(x0, q_past, q, u_past, u, u_next)
        return q_next
    
    def trajectory(self, u):
        x0 = self.init(u)
        u = np.array(u)
        n = int(len(x0)/2)
        traj = np.empty(shape = (self.N,n), dtype = np.float64)
        q_past, q = np.split(x0,2,0)
        traj[0], traj[1] = q_past, q
        for i in range(2, self.N, 1):
            q_past = traj[i-2]
            q = traj[i-1]
            q_next = self.get_root(q_past, q, u[i-2], u[i-1], u[i])
            traj[i] = q_next.T
            print('Iteration:{} '.format(i-1))
        q0 = traj[:-1]
        q1 = traj[1:]
        CD = self.f.Lagrangian_integration(q0, q1)
        coor = np.concatenate((CD[0], CD[1]),1)
        return traj, coor
  
    def init(self, u):
       x0, h = self.x0, self.h
       u = torch.tensor(u)
       _, q =generate_trajectory(self.model, x0, 1, h, u).get_trajectory()
       x = np.array(q)
       x = x[0,:2]
       x0 = np.concatenate([np.array(x0)[0:2],x],0)
       return x0
   
