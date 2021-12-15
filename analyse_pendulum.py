# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import matplotlib.pyplot as plt
import os, sys
import argparse
import torch

from Code.Utils import from_pickle
from Code.models import pendulum
from Code.integrate_models import implicit_integration_DEL, integrate_ODE
from Code.symo import SyMo_T
from Code.NN import LODE_T, NODE_T
from Code.models import get_field, pendulum

THIS_DIR = os.getcwd()


# %%
DPI = 300
FORMAT = 'png'
LINE_SEGMENTS = 10
ARROW_SCALE = 60
ARROW_WIDTH = 6e-3
LINE_WIDTH = 2
save_dir = "Experiments_pendulum/h=0.1"

def get_args():
    return {'fig_dir': './figures/pendulum/h=0.1',
            'gpu': 2,
            'pred_tol': 1e-5 ,
            'pred_maxiter': 10}

class ObjectView(object):
    def __init__(self, d): self.__dict__ = d

args = ObjectView(get_args())

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

def get_model(model, submodel, device):
    path = '{}/{}/{}{}-p-{}-stats.pkl'.format(THIS_DIR, save_dir, "pendulum-", model, submodel)
    stats = from_pickle(path)
    args = argparse.Namespace(**stats['hyperparameters'])
    
    if model == 'N-SyMo' or model== 'SyMo':
        nn = SyMo_T(args.num_angles, args.n_hidden_layers, args.n_neurons, args.time_step, args.nonlinearity).to(device).eval()
        path = '{}/{}/{}{}-p-{}.tar'.format(THIS_DIR, save_dir, "pendulum-", model, submodel)
        nn.load_state_dict(torch.load(path, map_location=device))
    
    elif 'L-NODE' in model:
        nn = LODE_T(args.num_angles, args.n_hidden_layers, args.n_neurons, args.nonlinearity).to(device).eval()
        path = '{}/{}/{}{}-p-{}.tar'.format(THIS_DIR, save_dir, "pendulum-", model, submodel)
        nn.load_state_dict(torch.load(path, map_location=device))

    elif 'NODE' in model:
        nn = NODE_T(args.num_angles, args.n_hidden_layers, args.n_neurons, args.nonlinearity, 0).to(device).eval()
        path = '{}/{}/{}{}-p-{}.tar'.format(THIS_DIR, save_dir, "pendulum-", model, submodel)
        nn.load_state_dict(torch.load(path, map_location=device))
    return nn, stats, args


# %%
models = ['8x32', '16x32', "32x32", "64x32", "128x32"]

def load_stats(nn, models):
    #loads the stats of all models
    train_loss = []
    test_loss= []
    int_loss = []
    int_std = []
    E_loss = []
    E_std = []
    H_loss = []
    H_std = []
    for model in models:
        path = '{}/{}/{}{}-p-{}-stats.pkl'.format(THIS_DIR, save_dir, "pendulum-", nn, model)
        stats = from_pickle(path)
        if 'SyMo' in nn:
            train_loss.append(stats['train_loss_poses'])
            test_loss.append(stats['test_loss_poses'])
        else:
            train_loss.append(stats['train_loss_poses'])
            test_loss.append(stats['test_loss_poses'])
        
        int_loss.append(stats['int_loss_poses'])
        int_std.append(stats['int_std'])
        E_loss.append(stats['E_loss'])
        E_std.append(stats['E_std'])
        if nn != 'NODE-rk4' and nn != "NODE-midpoint":
            H_loss.append(stats['H_loss'])
            H_std.append(stats['H_std'])
    if nn != 'NODE-rk4' and nn != "NODE-midpoint":
        return train_loss, test_loss, int_loss, int_std, E_loss, E_std, H_loss, H_std
    else:
        return train_loss, test_loss, int_loss, int_std, E_loss, E_std

#Load E2E-SyMo models
train_loss_N_SYMO, test_loss_N_SYMO, int_loss_N_SYMO, int_std_N_SYMO, E_loss_N_SYMO, E_std_N_SYMO, H_loss_N_SYMO, H_std_N_SYMO = load_stats('N-SyMo', models)
# Load SyMo models
train_loss_SYMO, test_loss_SYMO, int_loss_SYMO, int_std_SYMO, E_loss_SYMO, E_std_SYMO, H_loss_SYMO, H_std_SYMO = load_stats('SyMo', models)
#Load LODE_RK4 models
train_loss_LODE_RK4, test_loss_LODE_RK4, int_loss_LODE_RK4, int_std_LODE_RK4, E_loss_LODE_RK4, E_std_LODE_RK4, H_loss_LODE_RK4, H_std_LODE_RK4 = load_stats('L-NODE-rk4', models)
#Load LODE_RK2 models
train_loss_LODE_RK2, test_loss_LODE_RK2, int_loss_LODE_RK2, int_std_LODE_RK2, E_loss_LODE_RK2, E_std_LODE_RK2, H_loss_LODE_RK2, H_std_LODE_RK2 = load_stats('L-NODE-midpoint', models)
#Load NODE_RK4 models
train_loss_NODE_RK4, test_loss_NODE_RK4, int_loss_NODE_RK4, int_std_NODE_RK4, E_loss_NODE_RK4, E_std_NODE_RK4 = load_stats('NODE-rk4', models)
#Load NODE_RK2 models
train_loss_NODE_RK2, test_loss_NODE_RK2, int_loss_NODE_RK2, int_std_NODE_RK2, E_loss_NODE_RK2, E_std_NODE_RK2 = load_stats('NODE-midpoint', models)


# %%
x_axis = np.array([8, 16, 32, 64, 128])

fig = plt.figure(figsize=(18, 7), dpi=DPI)
plt.subplot(1, 3, 1)
plt.plot(x_axis, train_loss_NODE_RK4, 'bs-', label='NODE-rk4')
plt.plot(x_axis, train_loss_NODE_RK2, 'ms-', label='NODE-midpoint')
plt.plot(x_axis, train_loss_LODE_RK4, 'gs-', label= 'L-NODE-rk4')
plt.plot(x_axis, train_loss_LODE_RK2, 'ks-', label='L-NODE-midpoint')
plt.plot(x_axis, train_loss_SYMO, 'rs-', label='SyMo-midpoint')
plt.plot(x_axis, train_loss_N_SYMO, 'cs-', label  = 'E22-SyMo-midpoint')
plt.xscale('log')
plt.yscale('log')
# plt.xlabel('number of state initial condition')
plt.ylabel('Train error')
plt.xlabel('number of trajectories')
# plt.legend(fontsize=6)

plt.subplot(1, 3, 2)
plt.plot(x_axis, test_loss_NODE_RK4, 'bs-', label='NODE-rk4')
plt.plot(x_axis, test_loss_NODE_RK2, 'ms-', label='NODE-midpoint')
plt.plot(x_axis, test_loss_LODE_RK4, 'gs-', label= 'L-NODE-rk4')
plt.plot(x_axis, test_loss_LODE_RK2, 'ks-', label='L-NODE-midpoint')
plt.plot(x_axis, test_loss_SYMO, 'rs-', label='SyMo-midpoint')
plt.plot(x_axis, test_loss_N_SYMO, 'cs-', label  = 'E2E-SyMo-midpoint')
plt.xscale('log')
plt.yscale('log')
# plt.xlabel('number of state initial condition')
plt.xlabel('number of trajectories')
plt.ylabel('Test error')
# plt.legend(fontsize=6)

plt.subplot(1, 3, 3)
plt.plot(x_axis, int_loss_NODE_RK4, 'bs-', label='NODE-rk4')
plt.plot(x_axis, int_loss_NODE_RK2, 'ms-', label='NODE-midpoint')
plt.plot(x_axis, int_loss_LODE_RK4, 'gs-', label= 'L-NODE-rk4')
plt.plot(x_axis, int_loss_LODE_RK2, 'ks-', label='L-NODE-midpoint')
plt.plot(x_axis, int_loss_SYMO, 'rs-', label='SyMo-midpoint')
plt.plot(x_axis, int_loss_N_SYMO, 'cs-', label  = 'E2E-SyMo-midpoint')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('number of trajectories')
plt.ylabel('Integration error')
plt.legend(fontsize=8)

fig.savefig('{}/fig-train-pred-loss_pendulum.{}'.format(args.fig_dir, FORMAT))

fig = plt.figure(figsize=(16, 5), dpi=DPI)
plt.subplot(1, 2, 1)
plt.plot(x_axis, E_loss_NODE_RK4, 'bs-', label='NODE-rk4')
plt.plot(x_axis, E_loss_NODE_RK2, 'ms-', label='NODE-midpoint')
plt.plot(x_axis, E_loss_LODE_RK4, 'gs-', label= 'L-NODE-rk4')
plt.plot(x_axis, E_loss_LODE_RK2, 'ks-', label='L-NODE-midpoint')
plt.plot(x_axis, E_loss_SYMO, 'rs-', label='SyMo-midpoint', linewidth=2.1)
plt.plot(x_axis, E_loss_N_SYMO, 'cs-', label  = 'E2E-SyMo-midpoint')
plt.xscale('log')
plt.yscale('log')
plt.legend(fontsize=8)
# plt.xlabel('number of state initial condition')
plt.ylabel('Energy error')
plt.xlabel('number of trajectories')

plt.subplot(1, 2, 2)
plt.plot(x_axis, H_loss_LODE_RK4, 'gs-', label= 'L-NODE-rk4')
plt.plot(x_axis, H_loss_LODE_RK2, 'ks-', label='L-NODE-midpoint')
plt.plot(x_axis, H_loss_SYMO, 'rs-', label='SyMo-midpoint')
plt.plot(x_axis, H_loss_N_SYMO, 'cs-', label  = 'E2E-SyMo-midpoint')
plt.xscale('log')
plt.yscale('log')
# plt.xlabel('number of state initial condition')
plt.ylabel('Mass error')
plt.xlabel('number of trajectories')

fig.savefig('{}/fig-energy_inertia_pendulum.{}'.format(args.fig_dir, FORMAT))

# %% [markdown]
# # Analyse Moderate Regime Data Models

# %%
model = "32x32"
nn_symo, stats_symo, args_symo = get_model('SyMo', model, device)
nn_e2e_symo, stats_e2e_symo, args_e2e_symo = get_model('N-SyMo', model, device)
nn_lode_rk2, stats_lode_rk2, args_lode_rk2 = get_model('L-NODE-midpoint', model, device)
nn_lode_rk4, stats_lode_rk4, args_lode_rk4 = get_model('L-NODE-rk4', model, device)
nn_node_rk2, stats_node_rk2, args_node_rk2 = get_model('NODE-midpoint', model, device)
nn_node_rk4, stats_node_rk4, args_node_rk4 = get_model('NODE-rk4', model, device)

print('NODE-Midpoint')
print("Train Loss {:.4e}: Test Loss:{:.4e} Integration_loss: {:.4e} +/- {:.4e} Energy Loss:{:.4e} +/- {:.4e}".format(stats_node_rk2['train_loss_poses'], stats_node_rk2['test_loss_poses'], stats_node_rk2['int_loss_poses'], stats_node_rk2['int_std'], stats_node_rk2['E_loss'], stats_node_rk2['E_std']))
print('')
print('NODE-RK4')
print("Train Loss {:.4e}: Test Loss:{:.4e} Integration_loss: {:.4e} +/- {:.4e} Energy Loss:{:.4e} +/- {:.4e}".format(stats_node_rk4['train_loss_poses'], stats_node_rk4['test_loss_poses'], stats_node_rk4['int_loss_poses'], stats_node_rk4['int_std'], stats_node_rk4['E_loss'], stats_node_rk4['E_std']))
print('')
print('L-NODE-Midpoint')
print("Train Loss {:.4e}: Test Loss:{:.4e} Integration_loss: {:.4e} +/- {:.4e} Energy Loss:{:.4e} +/- {:.4e} Mass Loss:{:.4e} +/- {:.4e}".format(stats_lode_rk2['train_loss_poses'], stats_lode_rk2['test_loss_poses'], stats_lode_rk2['int_loss_poses'], stats_lode_rk2['int_std'], stats_lode_rk2['E_loss'], stats_lode_rk2['E_std'], stats_lode_rk2['H_loss'], stats_lode_rk2['H_std']))
print('')
print('L-NODE-RK4')
print("Train Loss {:.4e}: Test Loss:{:.4e} Integration_loss: {:.4e} +/- {:.4e} Energy Loss:{:.4e} +/- {:.4e} Mass Loss:{:.4e} +/- {:.4e}".format(stats_lode_rk4['train_loss_poses'], stats_lode_rk4['test_loss_poses'], stats_lode_rk4['int_loss_poses'], stats_lode_rk4['int_std'], stats_lode_rk4['E_loss'], stats_lode_rk4['E_std'], stats_lode_rk4['H_loss'], stats_lode_rk4['H_std']))
print('')
print('SyMo')
print("Train Loss {:.4e}: Test Loss:{:.4e} Integration_loss: {:.4e} +/- {:.4e} Energy Loss:{:.4e} +/- {:.4e} Mass Loss:{:.4e} +/- {:.4e}".format(stats_symo['train_loss_poses'], stats_symo['test_loss_poses'], stats_symo['int_loss_poses'], stats_symo['int_std'], stats_symo['E_loss'], stats_symo['E_std'], stats_symo['H_loss'], stats_symo['H_std']))
print('')
print('E2E-SyMo')
print("Train Loss {:.4e}: Test Loss:{:.4e} Integration_loss: {:.4e} +/- {:.4e} Energy Loss:{:.4e} +/- {:.4e} Mass Loss:{:.4e} +/- {:.4e}".format(stats_e2e_symo['train_loss_poses'], stats_e2e_symo['test_loss_poses'], stats_e2e_symo['int_loss_poses'], stats_e2e_symo['int_std'], stats_e2e_symo['E_loss'], stats_e2e_symo['E_std'], stats_e2e_symo['H_loss'], stats_e2e_symo['H_std']))
print('')


# %%
#get learning args
train_args = argparse.Namespace(**stats_symo['hyperparameters'])
#%% Training/Test losses and aditional information
fig = plt.figure(figsize=(15, 8), dpi=DPI)
plt.subplot(2, 2, 1)

plt.plot(stats_e2e_symo['train_losses'], 'r-', label='E2E-SyMo', linewidth=2)
plt.plot(stats_symo['train_losses'], 'r--', label='SyMo', linewidth=2)
plt.plot(stats_lode_rk2['train_losses'], 'b--', label='L-NODE-Midpoint', linewidth=2)
plt.plot(stats_lode_rk4['train_losses'], 'b-', label='L-NODE-RK4', linewidth=2)
plt.plot(stats_node_rk2['train_losses'], 'g--', label='NODE-Midpoint', linewidth=2)
plt.plot(stats_node_rk4['train_losses'], 'g-', label='NODE-RK4', linewidth=2)
plt.title("Training Losses", pad=14, fontsize=14)
plt.xlabel("Epoch", fontsize=14)
plt.ylabel("Loss", fontsize=14)
plt.yscale('log')
plt.subplot(2, 2, 2)

plt.plot(stats_e2e_symo['test_losses'], 'r-', label='E2E-SyMo', linewidth=2)
plt.plot(stats_symo['test_losses'], 'r--', label='SyMo', linewidth=2)
plt.plot(stats_lode_rk2['test_losses'], 'b--', label='L-NODE-Midpoint', linewidth=2)
plt.plot(stats_lode_rk4['test_losses'], 'b-', label='L-NODE-RK4', linewidth=2)
plt.plot(stats_node_rk2['test_losses'], 'g--', label='NODE-Midpoint', linewidth=2)
plt.plot(stats_node_rk4['test_losses'], 'g-', label='NODE-RK4', linewidth=2)
plt.title("Validation Losses", pad=14, fontsize=14)
plt.xlabel("Epoch", fontsize=14)
plt.ylabel("Loss", fontsize=14)
plt.yscale('log')

plt.subplot(2, 2, 3)
plt.plot(stats_e2e_symo['train_losses_poses'], 'r-', label='E2E-SyMo', linewidth=2)
plt.plot(stats_symo['train_losses_poses'], 'r--', label='SyMo', linewidth=2)
plt.plot(stats_lode_rk2['train_losses_poses'], 'b--', label='L-NODE-Midpoint', linewidth=2)
plt.plot(stats_lode_rk4['train_losses_poses'], 'b-', label='L-NODE-RK4', linewidth=2)
plt.plot(stats_node_rk2['train_losses_poses'], 'g--', label='NODE-Midpoint', linewidth=2)
plt.plot(stats_node_rk4['train_losses_poses'], 'g-', label='NODE-RK4', linewidth=2)
plt.title("Training Configuration Space Losses", pad=14, fontsize=14)
plt.xlabel("Epoch", fontsize=14)
plt.ylabel("Loss", fontsize=14)
plt.yscale('log')

plt.subplot(2, 2, 4)
plt.plot(stats_e2e_symo['test_losses_poses'], 'r-', label='E2E-SyMo', linewidth=2)
plt.plot(stats_symo['test_losses_poses'], 'r--', label='SyMo', linewidth=2)
plt.plot(stats_lode_rk2['test_losses_poses'], 'b--', label='L-NODE-Midpoint', linewidth=2)
plt.plot(stats_lode_rk4['test_losses_poses'], 'b-', label='L-NODE-RK4', linewidth=2)
plt.plot(stats_node_rk2['test_losses_poses'], 'g--', label='NODE-Midpoint', linewidth=2)
plt.plot(stats_node_rk4['test_losses_poses'], 'g-', label='NODE-RK4', linewidth=2)
plt.title("Validation Configuration Space Losses", pad=14, fontsize=14)
plt.xlabel("Epoch", fontsize=14)
plt.ylabel("Loss", fontsize=14)
plt.yscale('log')
fig.tight_layout()
plt.savefig('{}/losses_pendulum'.format(args.fig_dir, 'png'))


# %%
#%% Integrate models and plot
x0 = [np.pi/2, 0] 

h=args_symo.time_step #all models were trained under the same time_step
N=4000
u = np.zeros(shape=(N +2))
t_eval = np.linspace(0, (N+1)*h , N+2)
import gym 
import Code.myenv
#get ground truth
env = gym.make('MyPendulum-v0')
env.dt = args_symo.time_step
env.reset()
env.state = np.array(x0, dtype=np.float32)
obs = env._get_obs()
obs_list = []
obs_list.append(env.state)

for i in range(len(t_eval)):
    obs, _, _, _ = env.step([u[i]])
    obs_list.append(obs)
traj = np.stack(obs_list)

x_true = traj
y_true = traj[2:]

x0_odes = torch.tensor(x_true[1]).float().to(device)
x0_symos = torch.tensor(x_true[:2, 0]).float().to(device)

x_gt = torch.tensor(x_true[1:])
y_gt = torch.tensor(y_true[1:])

y_pred_node_midpoint = integrate_ODE(nn_node_rk2, "midpoint", x0_odes, N, h, device)
y_pred_node_rk4 = integrate_ODE(nn_node_rk4, "rk4", x0_odes, N, h, device)
y_pred_lode_midpoint = integrate_ODE(nn_lode_rk2, "midpoint", x0_odes, N, h, device)
y_pred_lode_rk4 = integrate_ODE(nn_lode_rk4, "rk4", x0_odes, N, h, device)

#Integrate models symos
y_pred_symo = implicit_integration_DEL(args_symo.root_find, N, h, nn_symo, x0_symos, args.pred_tol, args.pred_maxiter, device, us = None)
y_pred_e2e_symo = implicit_integration_DEL(args_e2e_symo.root_find, N, h, nn_e2e_symo, x0_symos, args.pred_tol, args.pred_maxiter, device, us = None)
#get true quantities
V_true = pendulum().potential_energy(y_gt)
T_true = pendulum().kinetic_energy(y_gt)
True_E = torch.ones_like(T_true)*pendulum().energy(np.array([x0]))
#Learned quantities
H_learned_lode_midpoint, V_learned_lode_midpoint, T_learned_lode_midpoint = nn_lode_rk2.get_matrices(y_pred_lode_midpoint) 
H_learned_lode_rk4, V_learned_lode_rk4, T_learned_lode_rk4 = nn_lode_rk4.get_matrices(y_pred_lode_rk4) 
H_learned_symo, V_learned_symo, T_learned_symo = nn_symo.get_matrices(y_pred_symo) 
H_learned_e2e_symo, V_learned_e2e_symo, T_learned_e2e_symo = nn_e2e_symo.get_matrices(y_pred_e2e_symo) 

E_learned_lode_midpoint = T_learned_lode_midpoint + V_learned_lode_midpoint
E_learned_lode_rk4 = T_learned_lode_rk4 + V_learned_lode_rk4
E_learned_symo = T_learned_symo + V_learned_symo
E_learned_e2e_symo = T_learned_e2e_symo + V_learned_e2e_symo

#Get learned momentums
config_space_symo = torch.cat((torch.tensor([[x0_symos[1], y_pred_symo[0,0]]]).to(device), torch.stack((y_pred_symo[:-1, 0], y_pred_symo[1:, 0]), 1)), 0)
config_space_e2e_symo = torch.cat((torch.tensor([[x0_symos[1], y_pred_symo[0,0]]]).to(device), torch.stack((y_pred_e2e_symo[:-1, 0], y_pred_e2e_symo[1:, 0]), 1)), 0)

phase_space_symo = nn_symo.momentum(config_space_symo.float().to(device)).cpu() 
phase_space_e2e_symo = nn_e2e_symo.momentum(config_space_e2e_symo.float().to(device)).cpu() 
phase_space_lode_midpoint = nn_lode_rk2.momentum( y_pred_lode_midpoint).cpu()
phase_space_lode_rk4 = nn_lode_rk4.momentum(y_pred_lode_rk4).cpu()
phase_space_node_midpoint =y_pred_node_midpoint.cpu()
phase_space_node_midpoint[:, 1] *=  (1/3)
phase_space_node_rk4 = y_pred_node_rk4.cpu()
phase_space_node_rk4[:, 1] *=  (1/3)

u=0
R = 9
gridsize = 25
kwargs = {'xmin': -R, 'xmax': R, 'ymin': -R, 'ymax': R, 'gridsize': gridsize, 'u': u}
field = get_field(**kwargs)

t_eval = t_eval[2:]


# %%
#%% Plot
fig = plt.figure(figsize=(15, 8), dpi=DPI)
plt.subplot(2, 3, 1)
plt.quiver(field['x'][:,0], (1/3)*field['x'][:,1], field['dx'][:,0], (1/3)*field['dx'][:,1],
            cmap='gray_r', scale=ARROW_SCALE, width=ARROW_WIDTH, color=(.2,.2,.2), alpha= 0.8) 

for i, l in enumerate(np.split(phase_space_e2e_symo, LINE_SEGMENTS)):
    color = (float(i)/LINE_SEGMENTS, 0, 1-float(i)/LINE_SEGMENTS)
    plt.plot(l[:,0], l[:,1],color=color, linewidth=LINE_WIDTH)

plt.xlabel("$q$", fontsize=14)
plt.ylabel("$p$", rotation=0, fontsize=14)
plt.title("E2E-SyMo", pad=10)
plt.xlim(0, 2*np.pi)
plt.ylim(-3, 3)

plt.subplot(2, 3, 2)

plt.quiver(field['x'][:,0], (1/3)*field['x'][:,1], field['dx'][:,0], (1/3)*field['dx'][:,1],
            cmap='gray_r', scale=ARROW_SCALE, width=ARROW_WIDTH, color=(.2,.2,.2), alpha = 0.8) 
for i, l in enumerate(np.split(phase_space_lode_rk4, LINE_SEGMENTS)):
    color = (float(i)/LINE_SEGMENTS, 0, 1-float(i)/LINE_SEGMENTS)
    plt.plot(l[:,0], l[:,1],color=color, linewidth=LINE_WIDTH)


plt.title("L-NODE-RK4", pad=10)
plt.xlim(0, 2*np.pi)
plt.ylim(-3, 3)
plt.xlabel("$q$", fontsize=14)
plt.ylabel("$p$", rotation=0, fontsize=14)
plt.subplot(2, 3, 3)

plt.quiver(field['x'][:,0], (1/3)*field['x'][:,1], field['dx'][:,0], (1/3)*field['dx'][:,1],
            cmap='gray_r', scale=ARROW_SCALE, width=ARROW_WIDTH, color=(.2,.2,.2), alpha = 0.8) 

for i, l in enumerate(np.split(phase_space_node_rk4, LINE_SEGMENTS)):
    color = (float(i)/LINE_SEGMENTS, 0, 1-float(i)/LINE_SEGMENTS)
    plt.plot(l[:,0], l[:,1],color=color, linewidth=LINE_WIDTH)

plt.title("NODE-RK4", pad=10)
plt.xlim(0, 2*np.pi)
plt.ylim(-3, 3)
plt.xlabel("$q$", fontsize=14)
plt.ylabel("$p$", rotation=0, fontsize=14)

plt.subplot(2, 3, 4)
plt.quiver(field['x'][:,0], (1/3)*field['x'][:,1], field['dx'][:,0], (1/3)*field['dx'][:,1],
            cmap='gray_r', scale=ARROW_SCALE, width=ARROW_WIDTH, color=(.2,.2,.2), alpha = 0.8) 

for i, l in enumerate(np.split(phase_space_symo, LINE_SEGMENTS)):
    color = (float(i)/LINE_SEGMENTS, 0, 1-float(i)/LINE_SEGMENTS)
    plt.plot(l[:,0], l[:,1],color=color, linewidth=LINE_WIDTH)

plt.title("SyMo", pad=10)
plt.xlim(0, 2*np.pi)
plt.ylim(-3, 3)
plt.xlabel("$q$", fontsize=14)
plt.ylabel("$p$", rotation=0, fontsize=14)

plt.subplot(2, 3, 5)

plt.quiver(field['x'][:,0], (1/3)*field['x'][:,1], field['dx'][:,0], (1/3)*field['dx'][:,1],
            cmap='gray_r', scale=ARROW_SCALE, width=ARROW_WIDTH, color=(.2,.2,.2), alpha = 0.8) 

for i, l in enumerate(np.split(phase_space_lode_midpoint, LINE_SEGMENTS)):
    color = (float(i)/LINE_SEGMENTS, 0, 1-float(i)/LINE_SEGMENTS)
    plt.plot(l[:,0], l[:,1],color=color, linewidth=LINE_WIDTH)

plt.title("L-NODE-Midpoint", pad=10)
plt.xlim(0, 2*np.pi)
plt.ylim(-3, 3)
plt.xlabel("$q$", fontsize=14)
plt.ylabel("$p$", rotation=0, fontsize=14)

plt.subplot(2, 3, 6)

plt.quiver(field['x'][:,0], (1/3)*field['x'][:,1], field['dx'][:,0], (1/3)*field['dx'][:,1],
            cmap='gray_r', scale=ARROW_SCALE, width=ARROW_WIDTH, color=(.2,.2,.2), alpha = 0.8)  
            
for i, l in enumerate(np.split(phase_space_node_midpoint, LINE_SEGMENTS)):
    color = (float(i)/LINE_SEGMENTS, 0, 1-float(i)/LINE_SEGMENTS)
    plt.plot(l[:,0], l[:,1],color=color, linewidth=LINE_WIDTH)

plt.title("NODE-Midpoint", pad=10)
plt.xlim(0, 2*np.pi)
plt.ylim(-3, 3)
plt.xlabel("$q$", fontsize=14)
plt.ylabel("$p$", rotation=0, fontsize=14)

fig.tight_layout()
plt.savefig('{}/Phase_spaces_pendulum'.format(args.fig_dir, 'png'))

# %% [markdown]
# ## Learned Quantities

# %%
#%% Plot learned quantities
fig = plt.figure(figsize=(16, 4), dpi=DPI)

plt.subplot(1, 4, 1)
H_true = torch.ones_like(H_learned_e2e_symo)*pendulum().H
n=100
plt.plot(t_eval[:n], H_true.squeeze(1).cpu()[:n], 'k-', label='Ground Truth', linewidth=2)
plt.plot(t_eval[:n], H_learned_e2e_symo.squeeze(1).cpu()[:n], 'r-', label='E2E-SyMo', linewidth=2)
plt.plot(t_eval[:n], H_learned_symo.squeeze(1).cpu()[:n], 'r--', label='SyMo', linewidth=2)
plt.plot(t_eval[:n], H_learned_lode_midpoint.squeeze(1).cpu()[:n], 'b--', label='L-NODE-Midpoint', linewidth=2)
plt.plot(t_eval[:n], H_learned_lode_rk4.squeeze(1).cpu()[:n], 'b-', label='L-NODE-RK4', linewidth=2)
plt.title("Inertia", pad=10)
plt.xlabel("Time [s]", fontsize=14)
plt.subplot(1, 4, 2)

plt.plot(t_eval[:n], T_true[:n], 'k-', label='Ground Truth', linewidth=2)
plt.plot(t_eval[:n], T_learned_e2e_symo.cpu()[:n], 'r-', label='E2E-SyMo', linewidth=2)
plt.plot(t_eval[:n], T_learned_symo.cpu()[:n], 'r--', label='SyMo', linewidth=2)
plt.plot(t_eval[:n], T_learned_lode_midpoint.cpu()[:n], 'b--', label='L-NODE-Midpoint', linewidth=2)
plt.plot(t_eval[:n], T_learned_lode_rk4.cpu()[:n], 'b-', label='L-NODE-RK4', linewidth=2)
plt.title("Kinetic Energy", pad=10)
plt.xlabel("Time [s]", fontsize=14)

plt.subplot(1, 4, 3)
plt.plot(t_eval[:n], V_true[:n], 'k-', label='Ground Truth', linewidth=2)
plt.plot(t_eval[:n], V_learned_e2e_symo.cpu()[:n], 'r-', label='E2E-SyMo', linewidth=2)
plt.plot(t_eval[:n], V_learned_symo.cpu()[:n], 'r--', label='SyMo', linewidth=2)
plt.plot(t_eval[:n], V_learned_lode_midpoint.cpu()[:n], 'b--', label='L-NODE-Midpoint', linewidth=2)
plt.plot(t_eval[:n], V_learned_lode_rk4.cpu()[:n], 'b-', label='L-NODE-RK4', linewidth=2)
plt.title("Potential Energy", pad=10)
plt.xlabel("Time [s]", fontsize=14)

plt.subplot(1, 4, 4)
plt.plot(t_eval[:n], True_E[:n], 'k-', label='Ground Truth', linewidth=2)
plt.plot(t_eval[:n], E_learned_e2e_symo.cpu()[:n], 'r-', label='E2E-SyMo', linewidth=2)
plt.plot(t_eval[:n], E_learned_symo.cpu()[:n], 'r--', label='SyMo', linewidth=2)
plt.plot(t_eval[:n], E_learned_lode_midpoint.cpu()[:n], 'b--', label='L-NODE-Midpoint', linewidth=2)
plt.plot(t_eval[:n], E_learned_lode_rk4.cpu()[:n], 'b-', label='L-NODE-RK4', linewidth=2)
plt.title("Total Energy", pad=10)
plt.xlabel("Time [s]", fontsize=14)

fig.tight_layout()
plt.legend(fontsize=7)
plt.savefig('{}/Learned_quantities_pendulum'.format(args.fig_dir, 'png'))


# %%
# %% Plot trajectory energies and MSE
#Energies
E_node_midpoint = pendulum().energy(y_pred_node_midpoint.cpu())
E_node_rk4 = pendulum().energy(y_pred_node_rk4.cpu())
E_lode_midpoint = pendulum().energy(y_pred_lode_midpoint.cpu())
E_lode_rk4 = pendulum().energy(y_pred_lode_rk4.cpu())
E_symo = pendulum().energy(y_pred_symo.cpu())
E_e2e_symo = pendulum().energy(y_pred_e2e_symo.cpu())

T_node_midpoint = pendulum().kinetic_energy(y_pred_node_midpoint.cpu())
T_node_rk4 = pendulum().kinetic_energy(y_pred_node_rk4.cpu())
T_lode_midpoint = pendulum().kinetic_energy(y_pred_lode_midpoint.cpu())
T_lode_rk4 = pendulum().kinetic_energy(y_pred_lode_rk4.cpu())
T_symo = pendulum().kinetic_energy(y_pred_symo.cpu())
T_e2e_symo = pendulum().kinetic_energy(y_pred_e2e_symo.cpu())

V_node_midpoint = pendulum().potential_energy(y_pred_node_midpoint.cpu())
V_node_rk4 = pendulum().potential_energy(y_pred_node_rk4.cpu())
V_lode_midpoint = pendulum().potential_energy(y_pred_lode_midpoint.cpu())
V_lode_rk4 = pendulum().potential_energy(y_pred_lode_rk4.cpu())
V_symo = pendulum().potential_energy(y_pred_symo.cpu())
V_e2e_symo = pendulum().potential_energy(y_pred_e2e_symo.cpu())

#MSE between configurations
MSE_node_midpoint = (y_gt[:, 0] - y_pred_node_midpoint[:, 0].cpu())**2
MSE_node_rk4 = (y_gt[:,0] - y_pred_node_rk4[:, 0].cpu())**2
MSE_lode_midpoint = (y_gt[:, 0] - y_pred_lode_midpoint[:, 0].cpu())**2
MSE_lode_rk4 = (y_gt[:, 0] - y_pred_lode_rk4[:, 0].cpu())**2
MSE_symo = (y_gt[:, 0] - y_pred_symo[:, 0].cpu())**2
MSE_e2e_symo = (y_gt[:, 0] - y_pred_e2e_symo[:, 0].cpu())**2

fig = plt.figure(figsize=(10, 4), dpi=DPI)
n=500
ax= plt.subplot(1,2,1)
plt.plot(t_eval[:n], True_E[:n], 'k-', label='Ground Truth', linewidth=2)
plt.plot(t_eval[:n], E_e2e_symo[:n], 'r-', label='E2E-SyMo', linewidth=2)
plt.plot(t_eval[:n], E_symo[:n], 'r--', label='SyMo', linewidth=2)
plt.plot(t_eval[:n], E_lode_midpoint[:n], 'b--', label='L-NODE-Midpoint', linewidth=2)
plt.plot(t_eval[:n], E_lode_rk4[:n], 'b-', label='L-NODE-RK4', linewidth=2)
plt.plot(t_eval[:n], E_node_midpoint[:n], 'g--', label='NODE-Midpoint', linewidth=2)
plt.plot(t_eval[:n], E_node_rk4[:n], 'g-', label='NODE-RK4', linewidth=2)
plt.title("Trajectory Energy", pad=10)
plt.ylim(-0.5, 20)
plt.xlabel("Time [s]", fontsize=14)
plt.legend(fontsize=7)

ax=plt.subplot(1,2,2)

plt.plot(t_eval[:n], MSE_e2e_symo[:n], 'r-', label='E2E-SyMo', linewidth=2)
plt.plot(t_eval[:n], MSE_symo[:n], 'r--', label='SyMo', linewidth=2)
plt.plot(t_eval[:n], MSE_lode_midpoint[:n], 'b--', label='L-NODE-Midpoint', linewidth=2)
plt.plot(t_eval[:n], MSE_lode_rk4[:n], 'b-', label='L-NODE-RK4', linewidth=2)
plt.plot(t_eval[:n], MSE_node_midpoint[:n], 'g--', label='NODE-Midpoint', linewidth=2)
plt.plot(t_eval[:n], MSE_node_rk4[:n], 'g-', label='NODE-RK4', linewidth=2)
plt.title("Configuration Space MSE", pad=10)
plt.xlabel("Time [s]", fontsize=14)
plt.ylim(-0.5, 20)

fig.tight_layout()
plt.savefig('{}/E_MSE_pendulum'.format(args.fig_dir, 'png'))

# %% [markdown]
# ## Forced Integration
# 

# %%
#%% Integrate models with forces
x0 = [np.pi/2, 0] 

h=args_symo.time_step #all models were trained under the same time_step
N=200
t_eval1 = np.linspace(0, (N+1)*h , N+2)
T = 10
u = np.sin(2*np.pi*t_eval1/T)
import gym 
import Code.myenv
#get ground truth
env = gym.make('MyPendulum-v0')
env.dt = h
env.reset()
env.state = np.array(x0, dtype=np.float32)
obs = env._get_obs()
obs_list = []

for i in range(len(t_eval1)):
    obs_list.append(obs)
    obs, _, _, _ = env.step([u[i]])
traj_forced = np.stack(obs_list)

x0_odes_forced = torch.tensor(traj_forced[1]).float().to(device)
x0_symos_forced = torch.tensor(traj_forced[:2, 0]).float().to(device)

u = np.expand_dims(u, 1)
y_pred_node_midpoint_forced = integrate_ODE(nn_node_rk2, "midpoint", x0_odes_forced, N, h, device, u[1:-1])
y_pred_node_rk4_forced = integrate_ODE(nn_node_rk4, "rk4", x0_odes_forced, N, h, device, u[1:-1])
y_pred_lode_midpoint_forced = integrate_ODE(nn_lode_rk2, "midpoint", x0_odes_forced, N, h, device, u[1:-1])
y_pred_lode_rk4_forced = integrate_ODE(nn_lode_rk4, "rk4", x0_odes_forced, N, h, device, u[1:-1])

#Integrate models symos
y_pred_symo_forced = implicit_integration_DEL(args_symo.root_find, N, h, nn_symo, x0_symos_forced, args.pred_tol, args.pred_maxiter, device, us = u)
y_pred_e2e_symo_forced = implicit_integration_DEL(args_e2e_symo.root_find, N, h, nn_e2e_symo, x0_symos_forced, args.pred_tol, args.pred_maxiter, device, us = u)

H_forced_lode_midpoint, V_learned_lode_midpoint, T_learned_lode_midpoint = nn_lode_rk2.get_matrices(y_pred_lode_midpoint_forced) 
H_forced_lode_rk4, V_learned_lode_rk4, T_learned_lode_rk4 = nn_lode_rk4.get_matrices(y_pred_lode_rk4_forced) 
H_forced_symo, V_learned_symo, T_learned_symo = nn_symo.get_matrices(y_pred_symo_forced) 
H_forced_e2e_symo, V_learned_e2e_symo, T_learned_e2e_symo = nn_e2e_symo.get_matrices(y_pred_e2e_symo_forced) 

y_pred_node_midpoint_forced = y_pred_node_midpoint_forced .cpu().numpy()
y_pred_node_rk4_forced = y_pred_node_rk4_forced.cpu().numpy()
y_pred_lode_midpoint_forced = y_pred_lode_midpoint_forced.cpu().numpy()
y_pred_lode_rk4_forced = y_pred_lode_rk4_forced.cpu().numpy()
y_pred_symo_forced = y_pred_symo_forced.cpu().numpy()
y_pred_e2e_symo_forced = y_pred_e2e_symo_forced.cpu().numpy()

y_true_forced = traj_forced[2:]
#MSE between configurations
MSE_node_midpoint_forced = (y_true_forced[:, 0] - y_pred_node_midpoint_forced[:, 0])**2
MSE_node_rk4_forced = (y_true_forced[:,0] - y_pred_node_rk4_forced[:, 0])**2
MSE_lode_midpoint_forced = (y_true_forced[:, 0] - y_pred_lode_midpoint_forced[:, 0])**2
MSE_lode_rk4_forced = (y_true_forced[:, 0] - y_pred_lode_rk4_forced[:, 0])**2
MSE_symo_forced = (y_true_forced[:, 0] - y_pred_symo_forced[:, 0])**2
MSE_e2e_symo_forced = (y_true_forced[:, 0] - y_pred_e2e_symo_forced[:, 0])**2

True_E_forced = pendulum().energy(y_true_forced)
E_node_midpoint_forced = pendulum().energy(y_pred_node_midpoint_forced)
E_node_rk4_forced = pendulum().energy(y_pred_node_rk4_forced)
E_lode_midpoint_forced = pendulum().energy(y_pred_lode_midpoint_forced)
E_lode_rk4_forced = pendulum().energy(y_pred_lode_rk4_forced)
E_symo_forced = pendulum().energy(y_pred_symo_forced)
E_e2e_symo_forced = pendulum().energy(y_pred_e2e_symo_forced)

MSE_E_e2e_symo = (True_E_forced - E_e2e_symo_forced)**2 
MSE_E_symo = (True_E_forced - E_symo_forced)**2
MSE_E_lode_midpoint = (True_E_forced - E_lode_midpoint_forced)**2
MSE_E_lode_rk4 = (True_E_forced - E_lode_rk4_forced)**2
MSE_E_node_midpoint = (True_E_forced - E_node_midpoint_forced)**2
MSE_E_node_rk4 = (True_E_forced - E_node_rk4_forced)**2


H_true = torch.ones_like(H_forced_e2e_symo)*pendulum().H


# %%
# %% plot forced results
from matplotlib import gridspec
t_eval = t_eval1[2:]
fig = plt.figure(figsize=(18, 6), dpi=DPI)
gs = gridspec.GridSpec(2, 6) 

plt.subplot(gs[0])
plt.plot(t_eval, y_true_forced[:, 0], 'k-', label='Ground Truth', linewidth=1)
plt.plot(t_eval, y_pred_e2e_symo_forced[:, 0], 'r-', label='E2E-SyMo', linewidth=1)
plt.ylabel('Trajectory')
plt.title('E2E-SyMo')

plt.subplot(gs[1])
plt.plot(t_eval, y_true_forced[:, 0], 'k-', label='Ground Truth', linewidth=1)
plt.plot(t_eval, y_pred_symo_forced[:, 0], 'r--', label='SyMo', linewidth=1)
plt.title('SyMo')

plt.subplot(gs[2])
plt.plot(t_eval, y_true_forced[:, 0], 'k-', label='Ground Truth', linewidth=1)
plt.plot(t_eval, y_pred_lode_midpoint_forced[:, 0], 'b--', label='L-NODE-Midpoint', linewidth=1)

plt.title('LODE-Midpoint')

plt.subplot(gs[3])
plt.plot(t_eval, y_true_forced[:, 0], 'k-', label='Ground Truth', linewidth=1)
plt.plot(t_eval, y_pred_lode_rk4_forced[:, 0], 'b-', label='L-NODE-RK4', linewidth=1)

plt.title('LODE-RK4')

plt.subplot(gs[4])
plt.plot(t_eval, y_true_forced[:, 0], 'k-', label='Ground Truth', linewidth=1)
plt.plot(t_eval, y_pred_node_midpoint_forced[:, 0], 'g--', label='NODE-Midpoint', linewidth=1)
plt.title('NODE-Midpoint')

plt.subplot(gs[5])
plt.plot(t_eval, y_true_forced[:, 0], 'k-', label='Ground Truth', linewidth=1)
plt.plot(t_eval, y_pred_node_rk4_forced[:, 0], 'g-', label='NODE-RK4', linewidth=1)
plt.title('NODE-RK4')

# Second axes
plt.subplot(gs[6])
ax1=plt.plot(t_eval, MSE_e2e_symo_forced, 'r-', label='E2E-SyMo', linewidth=1)
plt.xlabel('t')
plt.ylabel('MSE')

plt.subplot(gs[7])
ax2=plt.plot(t_eval, MSE_symo_forced, 'r--', label='SyMo', linewidth=1)
plt.xlabel('t')

plt.subplot(gs[8])
ax3=plt.plot(t_eval, MSE_lode_midpoint_forced, 'b--', label='L-NODE-Midpoint', linewidth=1)
plt.xlabel('t')

plt.subplot(gs[9])
ax4=plt.plot(t_eval, MSE_lode_rk4_forced, 'b-', label='L-NODE-RK4', linewidth=1)
plt.xlabel('t')

plt.subplot(gs[10])
ax5=plt.plot(t_eval, MSE_node_midpoint_forced, 'g--', label='NODE-Midpoint', linewidth=1)
plt.xlabel('t')

plt.subplot(gs[11])
ax6=plt.plot(t_eval, MSE_node_rk4_forced, 'g-', label='NODE-RK4', linewidth=1)
plt.xlabel('t')

fig.tight_layout()
plt.savefig('{}/traj_forced'.format(args.fig_dir, 'png'))

fig = plt.figure(figsize=(18, 6), dpi=DPI)
gs = gridspec.GridSpec(2, 6) 

plt.subplot(gs[0])
plt.plot(t_eval, True_E_forced, 'k-', label='Ground Truth', linewidth=1)
ax1=plt.plot(t_eval, E_e2e_symo_forced, 'r-', label='E2E-SyMo', linewidth=1)
plt.xlabel('t')
plt.ylabel('Energy')
plt.title('E2E-SyMo')

plt.subplot(gs[1])
plt.plot(t_eval, True_E_forced, 'k-', label='Ground Truth', linewidth=1)
ax2=plt.plot(t_eval, E_symo_forced, 'r--', label='SyMo', linewidth=1)
plt.xlabel('t')
plt.title('SyMo')

plt.subplot(gs[2])
plt.plot(t_eval, True_E_forced, 'k-', label='Ground Truth', linewidth=1)
ax3=plt.plot(t_eval, E_lode_midpoint_forced, 'b--', label='L-NODE-Midpoint', linewidth=1)
plt.xlabel('t')
plt.title('LODE-Midpoint')

plt.subplot(gs[3])
plt.plot(t_eval, True_E_forced, 'k-', label='Ground Truth', linewidth=1)
ax4=plt.plot(t_eval, E_lode_rk4_forced, 'b-', label='L-NODE-RK4', linewidth=1)
plt.xlabel('t')
plt.title('LODE-RK4')

plt.subplot(gs[4])
plt.plot(t_eval, True_E_forced, 'k-', label='Ground Truth', linewidth=1)
ax5=plt.plot(t_eval, E_node_midpoint_forced, 'g--', label='NODE-Midpoint', linewidth=1)
plt.xlabel('t')
plt.title('NODE-Midpoint')

plt.subplot(gs[5])
plt.plot(t_eval, True_E_forced, 'k-', label='Ground Truth', linewidth=1)
ax6=plt.plot(t_eval, E_node_rk4_forced, 'g-', label='NODE-RK4', linewidth=1)
plt.xlabel('t')
plt.title('NODE-RK4')

plt.subplot(gs[6])
plt.plot(t_eval, H_true.squeeze(1).cpu(), 'k-', label='Ground Truth', linewidth=1)
plt.plot(t_eval, H_forced_e2e_symo.squeeze(1).cpu(), 'r-', label='E2E-SyMo', linewidth=1)
plt.xlabel('t')
plt.ylabel('Inertia')
plt.ylim(0.3, 0.35)

plt.subplot(gs[7])
plt.plot(t_eval, H_true.squeeze(1).cpu(), 'k-', label='Ground Truth', linewidth=1)
plt.plot(t_eval, H_forced_symo.squeeze(1).cpu(), 'r--', label='SyMo', linewidth=1)
plt.xlabel('t')

plt.ylim(0.3, 0.35)

plt.subplot(gs[8])
plt.plot(t_eval, H_true.squeeze(1).cpu(), 'k-', label='Ground Truth', linewidth=1)
plt.plot(t_eval, H_forced_lode_midpoint.squeeze(1).cpu(), 'b--', label='L-NODE-Midpoint', linewidth=1)
plt.xlabel('t')
plt.ylim(0.3, 0.35)

plt.subplot(gs[9])
plt.plot(t_eval, H_true.squeeze(1).cpu(), 'k-', label='Ground Truth', linewidth=1)
plt.plot(t_eval, H_forced_lode_rk4.squeeze(1).cpu(), 'b-', label='L-NODE-RK4', linewidth=1)
plt.xlabel('t')
plt.ylim(0.3, 0.35)

ax=plt.subplot(gs[10:12])
ax.clear()  # clears the random data I plotted previously
ax.set_axis_off()  # removes the XY axes
from matplotlib.lines import Line2D
gt = Line2D([0], [0], color='black', linestyle='-', label='Ground Truth', linewidth=2)
e2e_symo = Line2D([0], [0], color='red', linestyle='-', label='E2E-SyMo', linewidth=2)
symo = Line2D([0], [0], color='red', linestyle='--', label='E2E-SyMo', linewidth=2)
lode_rk4 = Line2D([0], [0], color='blue', linestyle='-', label='L-NODE-RK4', linewidth=2)
lode_rk2 = Line2D([0], [0], color='blue', linestyle='--', label='L-NODE-Midpoint', linewidth=2)
node_rk4 = Line2D([0], [0], color='green', linestyle='-', label='NODE-RK4', linewidth=2)
node_rk2 = Line2D([0], [0], color='green', linestyle='--', label='NODE-Midpoint', linewidth=2)

handles=[ax1, ax2, ax3, ax4, ax5, ax6]
ax.legend(handles=[gt, e2e_symo, symo, lode_rk4, lode_rk2, node_rk4, node_rk2], loc="center")

fig.tight_layout()
plt.savefig('{}/energy_inertia_forced'.format(args.fig_dir, 'png'))

# %% [markdown]
# # Analyse Noise

# %%
models = ["32x32_sigma_0.01", "32x32_sigma_0.05", "32x32_sigma_0.1"] 
save_dir = "Experiments_pendulum/h=0.1_noise"

train_loss_N_SYMO, test_loss_N_SYMO, int_loss_N_SYMO, int_std_N_SYMO, E_loss_N_SYMO, E_std_N_SYMO, H_loss_N_SYMO, H_std_N_SYMO = load_stats('N-SyMo', models)
train_loss_SYMO, test_loss_SYMO, int_loss_SYMO, int_std_SYMO, E_loss_SYMO, E_std_SYMO, H_loss_SYMO, H_std_SYMO = load_stats('SyMo', models)
train_loss_LODE_RK4, test_loss_LODE_RK4, int_loss_LODE_RK4, int_std_LODE_RK4, E_loss_LODE_RK4, E_std_LODE_RK4, H_loss_LODE_RK4, H_std_LODE_RK4 = load_stats('L-NODE-rk4', models)
train_loss_LODE_RK2, test_loss_LODE_RK2, int_loss_LODE_RK2, int_std_LODE_RK2, E_loss_LODE_RK2, E_std_LODE_RK2, H_loss_LODE_RK2, H_std_LODE_RK2 = load_stats('L-NODE-midpoint', models)
train_loss_NODE_RK4, test_loss_NODE_RK4, int_loss_NODE_RK4, int_std_NODE_RK4, E_loss_NODE_RK4, E_std_NODE_RK4 = load_stats('NODE-rk4', models)
train_loss_NODE_RK2, test_loss_NODE_RK2, int_loss_NODE_RK2, int_std_NODE_RK2, E_loss_NODE_RK2, E_std_NODE_RK2 = load_stats('NODE-midpoint', models)


