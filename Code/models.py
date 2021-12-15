#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Used only for the analysis section - simulation is performed by the modified gym environments
"""
Created on Fri Jan 29 20:07:32 2021

@author: saul
"""

from abc import ABC, abstractmethod
import numpy as np
from numpy import sin, cos, pi


def get_field(xmin, xmax, ymin, ymax, gridsize, u=0):
    field = {'meta': locals()}

    # meshgrid to get vector field
    b, a = np.meshgrid(np.linspace(xmin, xmax, gridsize), np.linspace(ymin, ymax, gridsize))
    ys = np.stack([b.flatten(), a.flatten()])
    
    # get vector directions
    dydt = [pendulum().Lagrangian_analytical('None', y, u) for y in ys.T]
    dydt = np.stack(dydt).T
    field['x'] = ys.T
    field['dx'] = dydt.T
    return field




# Modified from OpenAI gym Acrobot, cartpole and pendulum environments





class abstract_model(object):
    
    @abstractmethod
    def Lagrangian_analytical(self, *args):
        "Returns forward dynamics"
    
    @abstractmethod  
    def kinetic_energy(self):
        "Return kinetic Energy"
    
    @abstractmethod  
    def potential_energy(self):
        "Return Potential Energy"
          
    def _energy(self, state):
        return self.potential_energy(state) + self.kinetic_energy(state)
    
        
class cartpole(abstract_model):
        def __init__(self):
            self.gravity = 9.81
            self.masscart = 1
            self.masspole = 0.1
            self.total_mass = (self.masspole + self.masscart)
            self.length = 0.5 # actually half the pole's length
            self.polemass_length = (self.masspole * self.length)
        def Lagrangian_analytical(self, *args):
            x, theta, x_dot, theta_dot = np.split(args[1], 4, 0)
            if args[2] is not None:
                 u = args[2]
            else:
                u = 0 
            force = u
            costheta = np.cos(theta)
            sintheta = np.sin(theta)
            temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
            thetaacc = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass))
            xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
            return np.concatenate((x_dot, theta_dot, xacc, thetaacc), 0)
            
        def kinetic_energy(self, state):
            x, t, dx, dt = state.T
            T = 0.5*self.masscart*(dx**2) + 0.5*self.masspole*((dx**2) + 2*self.length*dt*dx*np.cos(t) + (self.length**2)*(dt**2)) + (1/6)*(self.length**2)*self.masspole*(dt**2) 
            return T
        
        def potential_energy(self, state):
            x, t, dx, dt = state.T
            V = self.masspole*self.gravity*self.length*np.cos(t)
            return V
        
        def energy(self, state):
            return self._energy(state)
        
        def Inertia_matrix(self, state):
            x, t, dx, dt = state.T
            H12 = self.masspole*self.length*np.cos(t)
            H21 = self.masspole*self.length*np.cos(t)
            H22 = (4/3)*self.masspole*(self.length**2)*np.ones_like(H12)
            H11 = (self.masscart + self.masspole)*np.ones_like(H12)
            return np.stack((H11, H12, H21, H22),1)
    
    
class acrobot(abstract_model):
        def __init__(self):
            self.LINK_LENGTH_1 = 1.  # [m]
            self.LINK_LENGTH_2 = 1.  # [m]
            self.LINK_MASS_1 = 1.  #: [kg] mass of link 1
            self.LINK_MASS_2 = 1.  #: [kg] mass of link 2
            self.LINK_COM_POS_1 = 0.5  #: [m] position of the center of mass of link 1
            self.LINK_COM_POS_2 = 0.5  #: [m] position of the center of mass of link 2
            self.LINK_MOI = 1.  #: moments of inertia for both links
        def Lagrangian_analytical(self, *args):
            if args:
                theta1, theta2, dtheta1, dtheta2 = np.split(np.array(args[1]), 4, 0)
                
                if args[2] is not None:
                    u = np.array(args[2])
                else:
                    u = 0
            m1 = self.LINK_MASS_1
            m2 = self.LINK_MASS_2
            l1 = self.LINK_LENGTH_1
            lc1 = self.LINK_COM_POS_1
            lc2 = self.LINK_COM_POS_2
            I1 = self.LINK_MOI
            I2 = self.LINK_MOI
            g = 9.81
            
            
            d1 = m1 * lc1 ** 2 + m2 * \
                (l1 ** 2 + lc2 ** 2 + 2 * l1 * lc2 * cos(theta2)) + I1 + I2
            d2 = m2 * (lc2 ** 2 + l1 * lc2 * cos(theta2)) + I2
            phi2 = m2 * lc2 * g * cos(theta1 + theta2 - pi / 2.)
            phi1 = - m2 * l1 * lc2 * dtheta2 ** 2 * sin(theta2) \
                   - 2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * sin(theta2)  \
                + (m1 * lc1 + m2 * l1) * g * cos(theta1 - pi / 2) + phi2
            ddtheta2 = (u + d2 / d1 * phi1 - m2 * l1 * lc2 * dtheta1 ** 2 * np.sin(theta2) - phi2) \
                / (m2 * lc2 ** 2 + I2 - d2 ** 2 / d1)
            ddtheta1 = -(d2 * ddtheta2 + phi1) / d1
          
            
            return np.concatenate((dtheta1, dtheta2, ddtheta1, ddtheta2) ,0)
        
        def Inertia_matrix(self, state):
            m1 = self.LINK_MASS_1
            m2 = self.LINK_MASS_2
            l1 = self.LINK_LENGTH_1
            lc1 = self.LINK_COM_POS_1
            lc2 = self.LINK_COM_POS_2
            I1 = self.LINK_MOI
            I2 = self.LINK_MOI
            t1, t2, dt1, dt2 = state.T
            H11 = m1*(lc1**2) + m2*((l1**2) + (lc2**2) + 2*l1*lc2*np.cos(t2)) + I1 + I2
            H12 = m2*(lc2**2 + l1*lc2*np.cos(t2)) + I2
            H21 = H12
            H22 = m2*lc2**2 + I2*np.ones_like(H21)
            return np.stack((H11, H12, H21, H22),1)
        
        def kinetic_energy(self, state):
            t1, t2, dt1, dt2 = np.hsplit(state, 4)
            H11, H12, H21, H22 = np.hsplit(self.Inertia_matrix(state), 4)
            
            T1 = dt1*(dt1*H11 + dt2*H21)
            T2 = dt2*(dt1*H12 + dt2*H22)
            return (T1 + T2)*0.5
            
        
        def potential_energy(self, state):
            m1 = self.LINK_MASS_1
            m2 = self.LINK_MASS_2
            l1 = self.LINK_LENGTH_1
            l2 = self.LINK_LENGTH_2
            lc1 = self.LINK_COM_POS_1
            lc2 = self.LINK_COM_POS_2
            g=9.81
            t1, t2, dt1, dt2 = np.hsplit(state, 4)
            V = -m1*g*lc1*np.cos(t1) - m2*g*(l1*np.cos(t1) + lc2*np.cos(t1+t2)) 
            return V
        
        def energy(self, state):
            return self._energy(state)
        
class double_pendulum(abstract_model):
        """
        Double Pendulum simulator via Lagrangian and Hamiltonean formulation
    
        """
        def __init__(self, state, u=None):
            """
            Parameters
            ----------
            state : torch.tensor 1x4
                angle and respective accelerations
            formulation : string
                Hamiltonian or Lagrangian Formulation

            Returns
            -------
            None.

            """
            self.g = 9.8
            self.m1 = 1 
            self.m2 = 1
            self.l1 = 1
            self.l2 = 1
            self.state = np.array(state)
        
        def Lagrangian_analytical(self, *args):
            continuous = True
            if args:
                t1, t2, w1, w2, = np.split(np.array(args[0]), 4 , 0)
                continuous=False
            else:
                t1, t2, w1, w2 = self.state.T
            a1 = (self.l2 / self.l1) * (self.m2 / (self.m1 + self.m2)) * np.cos(t1 - t2)
            a2 = (self.l1 / self.l2) * np.cos(t1 - t2)
            f1 = -(self.l2 / self.l1) * (self.m2 / (self.m1 + self.m2)) * (w2**2) * np.sin(t1 - t2) - (self.g /self. l1) * np.sin(t1)
            f2 = (self.l1 / self.l2) * (w1**2) * np.sin(t1 - t2) - (self.g / self.l2) * np.sin(t2)
            g1 = (f1 - a1 * f2) / (1 - a1 * a2)
            g2 = (f2 - a2 * f1) / (1 - a1 * a2)
            
            if continuous:
                return  np.stack((g1, g2),1)
            else:    
                return np.concatenate((w1, w2, g1, g2) ,0)
        
        
        def kinetic_energy(self):
            t1, t2, w1, w2 = self.state.T
            T1 = 0.5 * self.m1 * (self.l1 * w1)**2
            T2 = 0.5 * self.m2 * ((self.l1 * w1)**2 + (self.l2 * w2)**2 + 2 *self. l1 * self.l2 * w1 * w2 * np.cos(t1 - t2))
            T = T1 + T2
            return T
        
        def potential_energy(self):
            t1, t2, w1, w2 = self.state.T
            y1 = -self.l1 * np.cos(t1)
            y2 = y1 - self.l2 * np.cos(t2)
            V = self.m1 * self.g * y1 + self.m2 * self.g * y2
            return V
        
        def energy(self):
            return self._energy()
        
        
        
class pendulum(abstract_model):
        """
        Pendulum simulation based on Open AI gym dynamics
    
        """
        def __init__(self):
            """
            Respects open Ai gym simulation
            
            """
            
            self.g = 9.81
            self.m = 1.
            self.l = 1.
            self.H = (1/3)*self.m*self.l**2
        def Lagrangian_analytical(self, *args):
            th, thdot = np.split(np.array(args[1]), 2, 0)
            
            if args[2] is not None:
                u = np.array(args[2])
            else:
                u = 0
                
            g = self.g
            m = self.m
            l = self.l    
            ddt = (-3 * g / (2 * l) * np.sin(th+np.pi) + 3. / (m * l ** 2) * u)
            
            return np.concatenate((thdot, ddt) ,0)
        
        def kinetic_energy(self, state):
            th, thdot = state.T
            Ig = (1/3)*self.m*self.l**2
            return 0.5*Ig*(thdot**2)
        
        def potential_energy(self, state):
            th, thdot = state.T
            V = 0.5*self.m*self.l*self.g*(1+cos(th))
            return V
        
        def energy(self, state):
            return self._energy(state)


