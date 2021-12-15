# code borrowed from Sam Greydanus
# https://github.com/greydanus/hamiltonian-nn

import sys,os

from numpy.core.fromnumeric import size
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)



import numpy as np
from Code.Utils import to_pickle, from_pickle
import gym

def sample_gym(seed=0, timesteps=10, h = 0.05, trials=50, 
              verbose=False, u =np.zeros(shape=(50, 10))):
    
    env_name='MyCartPole-v0'
    gym_settings = locals()
    if verbose:
        print("Making a dataset of Cartpole observations.")
    env: gym.wrappers.time_limit.TimeLimit = gym.make(env_name)
    env.seed(seed)
    env.dt = h

    trajs = []
    x0 = [] #Initial conditions
    for trial in range(trials):
        valid = False
        while not valid:
            env.reset()
            x = env.state
            traj = []
            for step in range(timesteps +2):   # +2 bcs symos requires information at more time steps
                obs, _, _, _ = env.step([u[trial, step]]) # action
                traj.append(obs)
            traj = np.stack(traj)
            if np.amax(traj[:, 0]) < 2*env.x_threshold - 0.001  and np.amin(traj[:, 0]) > -2*env.x_threshold + 0.001:
                if np.amax(traj[:, 3]) < env.MAX_VEL - 0.001  and np.amin(traj[:, 3]) > -env.MAX_VEL + 0.001:
                    valid = True
                    x0.append(x)
        trajs.append(traj)
    trajs = np.stack(trajs) # (trials, timesteps, 2)
    return trajs, np.stack(x0)

def get_dataset(seed=0, time_steps=10, samples=50, h= 0.05, samples_int=100, test_split=0.5, noise_std=None, us=[-1, 1], u_shape= "constant"):

    if u_shape == 'constant':
        np.random.seed(seed=seed)
        u_factor = np.random.uniform(us[0], us[1], (samples, 1)) 
        u = np.ones(shape=(samples, time_steps+ 2))*u_factor
    elif u_shape == 'zeros':
        u = np.zeros(shape=(samples, time_steps +2))

    trajs, x0 = sample_gym(seed=seed, timesteps=time_steps, h=h, trials=samples, u=u)
    trajs = np.float32(trajs)
    u = np.float32(u)
    if noise_std is not None:
        np.random.seed(seed=seed)
        trajs += np.random.randn(*trajs.shape)*noise_std
        u += np.random.randn(*u.shape)*noise_std

    # make a train/test split
    split_ix = int(samples * test_split)
    split_data = {}
    split_controls = {}
    split_data['train_x'], split_data['test_x'] = trajs[:split_ix,:], trajs[split_ix:,:]
    split_controls['train_u'], split_controls['test_u'] = u[:split_ix,:], u[split_ix:,:]
    data = split_data
    data['x0'] = x0
    controls = split_controls

    return data, controls


def arrange_DEL_dataset(data, u):
    u=np.expand_dims(u, 2)
    d_f = 2
    zero_controls = np.zeros_like(u)
    
    x = np.concatenate((data[:,:-2, :d_f], data[:, 1:-1, :d_f], u[:, :-2], zero_controls[:, :-2],  u[:, 1:-1], zero_controls[:, 1:-1],u[:, 2:], zero_controls[:, 2:]),2)
    y = data[:, 2:]
    return x, y, u

def arrange_NODE_dataset(data, u):
    x = data[:, 1:-1]
    y = data[:, 2:]
    u=np.expand_dims(u, 2)
    u = u[:, 1:-1]
    return x, y, u
