import torch, argparse
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pickle
import scipy.integrate
solve_ivp = scipy.integrate.solve_ivp

import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

from Code.Trainer import train
from Code.symo import SyMo_RT
from Code.Rigid_Body import MLP
from Code.Utils import get_n_params, to_pickle, from_pickle
from Code.integrate_models import integrate_ODE, implicit_integration_DEL
from Code.models import cartpole
from Code.Trainer import train
from Code.Newton import Newton
from Code.broyden import broyden
from Code.root_find import rootfind
from Code.NN import LODE_RT, NODE_RT, ODE
from Code.data import data

def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--model', type=str, default='E2E-SyMo', help="one of 'NODE', 'LODE', 'SyMo' and 'E2E-SyMo'")
    parser.add_argument('--name', default='cartpole', type=str, help='only one option right now')
    parser.add_argument('--num_angles', default=1, type=int, help='number of rotational coordinates')
    parser.add_argument('--num_trans', default=1, type=int, help='number of translational coordinates')
    parser.add_argument('--embedding', default=True, type=bool, help='If embedding is desired')
    parser.add_argument('--N_train', type=int, default='25', help="Number of different train trajectories")
    parser.add_argument('--n_train', type=int, default='32', help="Number of train observations in each trajectory")
    parser.add_argument('--N_test', type=int, default='128', help="Number of different test trajectories")
    parser.add_argument('--n_test', type=int, default='32', help="Number of test observations in each trajectory")
    parser.add_argument('--N_int', type=int, default='32', help="Number of different test trajectories")
    parser.add_argument('--n_int', type=int, default='2000', help="Number of test observations in each trajectory")
    parser.add_argument('--noise_std', type=float, default='0.', help="Induced noise std.")
    parser.add_argument('--train_seed', default=0, type=int, help=' training random seed')
    parser.add_argument('--noise_seed', default=0, type=int, help=' noise random seed')
    parser.add_argument('--int_seed', default=1, type=int, help=' integration random seed')
    parser.add_argument('--learning_rate', default=1e-3, type=float, help=' initial learning rate')
    parser.add_argument('--nonlinearity', default='tanh', type=str, help='neural net nonlinearity')
    parser.add_argument('--n_hidden_layers', default=2, type=int, help='number of hidden layers')
    parser.add_argument('--n_neurons', default=256, type=int, help='number of neurons')
    parser.add_argument('--n_epochs', default=2000, type=int, help='number of epochs')
    parser.add_argument('--patience', default=50, type=int, help='scheduler patience')
    parser.add_argument('--factor', default=0.7, type=float, help='scheduler patience')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--weight_decay', default=0, type=int, help='weight decay')
    parser.add_argument('--time_step', default=0.01, type=float, help='time_step')
    parser.add_argument('--root_find', default='Newton', type=str, help="one of 'Newton' or 'Broyden'")
    parser.add_argument('--odeint', default="rk4", type=str, help="One of the solvers available at https://github.com/rtqichen/torchdiffeq")
    parser.add_argument('--forward_tol', default=1e-5, type=float, help="If E2E-SyMo specify a forward tolerance")
    parser.add_argument('--backward_tol', default=1e-8, type=float, help="If Broyden specify a backward tolerance")
    parser.add_argument('--forward_maxiter', default=20, type=int, help="If E2E-SyMo specify the maximum number of iterations during forward pass")
    parser.add_argument('--backward_maxiter', default=20, type=int, help="If Broyden specify the maximum number of iterations during backward pass")
    parser.add_argument('--int_tol', default=1e-4, type=float, help="Integration tolerance for Symos")
    parser.add_argument('--int_maxiter', default=20, type=int, help="Integration maxiter for Symos")
    parser.add_argument('--save_dir', default=THIS_DIR, type=str, help='where to save the trained model')
    parser.add_argument('--gpu', type=int, default=2)
    parser.set_defaults(feature=True)
    return parser.parse_args() 

def total_loss(args, model, x, target):
    model = model.eval()
    q_past, q, u_past, u, u_next =  torch.split(x, args.d_f, 1)
    q_next = model(x.to(args.device))
    v_next = (q_next - q.to(args.device))/args.time_step
    output = torch.cat((q_next, v_next),1)
    criterion = torch.nn.MSELoss()
    return criterion(output, target.to(args.device))

def get_model(args, nn, model, trainloader, testloader, criterion):
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay) 
    scheduler = ReduceLROnPlateau(optimizer,  'min', patience = args.patience, verbose=True, factor=args.factor)
    return train(args.model, nn, model, criterion, optimizer, scheduler, args.device, trainloader, testloader, args.n_epochs)

def Train(args, x_train, y_train, u_train, x_test, y_test, u_test):

    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
    # reproducibility: set random seed
    args.device = device
    torch.manual_seed(230)
    np.random.seed(230)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    n_angles = args.num_angles
    n_trans = args.num_trans
    lr = args.learning_rate
    n_epochs = args.n_epochs
    activation = args.nonlinearity
    n_hidden_layers = args.n_hidden_layers
    n_neurons = args.n_neurons
    h = args.time_step
    embedding = args.embedding
    batch_size = args.batch_size
    weight_decay = args.weight_decay
    args.device = device
    args.d_f = args.num_angles + args.num_trans

    stats = {}


    #flatten the data
    x_train = torch.flatten(x_train, 0,1)
    y_train = torch.flatten(y_train, 0,1)
    x_test = torch.flatten(x_test, 0,1)
    y_test = torch.flatten(y_test, 0,1)
    u_train = torch.flatten(u_train, 0,1)
    u_test = torch.flatten(u_test, 0,1)
    
    
    #Create itegration data
    u_int_shape = torch.zeros(size=(1, args.n_int+2)) 
    data_int= data(args.name, x0_max, x0_min, h, args.N_int,0, args.n_int, args.int_seed, u_int_shape)
    x0, _ = data_int.random_initial_condition()
    E_int_true = cartpole().energy(x0)



    if (args.model == 'E2E-SyMo') and args.root_find == "Newton":
        args.input_dim = 3
        forward_tol = args.forward_tol
        forward_maxiter = args.forward_maxiter
        nn = SyMo_RT(n_trans, n_angles, n_hidden_layers, n_neurons, h, activation, embedding=embedding, ln=False).to(device)
        model = rootfind(nn, args.root_find, forward_tol, forward_maxiter)

        
        criterion = nn.implicit_loss
        train_data = TensorDataset(x_train, y_train[:, :args.d_f])
        trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_data = TensorDataset(x_test, y_test[:, :args.d_f])
        testloader = DataLoader(test_data, batch_size=len(x_test), shuffle=False)
        checkpoint = get_model(args, nn, model, trainloader, testloader, criterion)

    elif (args.model == 'E2E-SyMo') and args.root_find == "Broyden":
        args.input_dim = 3
        forward_tol = args.forward_tol
        forward_maxiter = args.forward_maxiter
        backward_tol = args.backward_tol
        backward_maxiter = args.backward_maxiter
        nn = SyMo_RT(n_trans, n_angles, n_hidden_layers, n_neurons, h, activation, embedding=embedding, ln=False).to(device)
        model = rootfind(nn, args.root_find, forward_tol, forward_maxiter, backward_tol, backward_maxiter)

        
        criterion = nn.implicit_loss
        train_data = TensorDataset(x_train, y_train[:, :args.d_f])
        trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
        test_data = TensorDataset(x_test, y_test[:, :args.d_f])
        testloader = DataLoader(test_data, batch_size=len(x_test), shuffle=False)
        checkpoint = get_model(args, nn, model, trainloader, testloader, criterion)

    
    
    elif (args.model == 'SyMo'):
        args.input_dim = 3
        forward_tol = args.forward_tol
        forward_maxiter = args.forward_maxiter
        nn = SyMo_RT(n_trans, n_angles, n_hidden_layers, n_neurons, h, activation, embedding=embedding, ln=False).to(device)
        model = rootfind(nn, args.root_find, forward_tol, forward_maxiter)
        criterion = nn.loss
        d_f = args.d_f

        #add q_next to the input data
        x_train_wq = torch.cat((x_train[:, :int(2*d_f)], y_train[:, :d_f], x_train[:, int(2*d_f):]), dim=1)
        x_test_wq = torch.cat((x_test[:, :int(2*d_f)], y_test[:, :d_f], x_test[:, int(2*d_f):]), dim=1)
        
        train_data = TensorDataset(x_train_wq, y_train[:, :args.d_f])
        trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
        test_data = TensorDataset(x_test_wq, y_test[:, :args.d_f])
        testloader = DataLoader(test_data, batch_size=len(x_test), shuffle=False)
        checkpoint = get_model(args, nn, model, trainloader, testloader, criterion)




    elif (args.model == 'LODE'):
        args.input_dim = 3
        nn = LODE_RT(n_trans, n_angles, n_hidden_layers, n_neurons, activation, embedding=embedding, ln=False).to(device)
        model = ODE(nn,args.odeint, h).to(device)
        criterion = nn.loss
        zero_controls = torch.zeros_like(u_train)
        x_train = torch.cat((x_train, u_train, zero_controls), dim=1)
        train_data = TensorDataset(x_train, y_train)
        trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
        zero_controls = torch.zeros_like(u_test)
        x_test = torch.cat((x_test, u_test, zero_controls), dim=1)
        test_data = TensorDataset(x_test, y_test)
        testloader = DataLoader(test_data, batch_size=len(x_test), shuffle=False)
        checkpoint = get_model(args, nn, model, trainloader, testloader, criterion)
        
    
    elif (args.model == 'NODE'):
        args.input_dim = 5
        u_index = 0
        nn = NODE_RT(n_trans, n_angles, n_hidden_layers, n_neurons, activation, u_index, embedding=embedding, ln=False).to(device)
        model = ODE(nn,args.odeint, h).to(device)
        criterion = nn.loss
        zero_controls = torch.zeros_like(u_train)
        x_train = torch.cat((x_train, u_train, zero_controls), dim=1)
        train_data = TensorDataset(x_train, y_train)
        trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
        zero_controls = torch.zeros_like(u_test)
        x_test = torch.cat((x_test, u_test, zero_controls), dim=1)
        test_data = TensorDataset(x_test, y_test)
        testloader = DataLoader(test_data, batch_size=len(x_test), shuffle=False)
        checkpoint = get_model(args, nn, model, trainloader, testloader, criterion)
        
    else:
        raise ValueError('model_type not supported')
    

    stats['train_loss'] = checkpoint['train_loss']
    stats['train_losses'] = checkpoint['train_losses']
    stats['test_loss'] = checkpoint['test_loss']
    stats['test_losses'] = checkpoint['test_losses']
    stats['train_loss_poses'] = checkpoint['train_loss_poses'] 
    stats['test_loss_poses'] = checkpoint['test_loss_poses'] 
    stats['train_losses_poses'] = checkpoint['train_losses_poses'] 
    stats['test_losses_poses'] = checkpoint['test_losses_poses'] 
    stats['optimizer_state_dict'] = checkpoint['optimizer_state_dict']

    #Integrate, compute energies, masses and respective losses 

    y_int_pred = torch.zeros(size=(args.N_int, args.n_int, int(2*args.d_f)))
    E_loss_per_traj = []
    Energies = torch.empty(size=(args.N_int, args.n_int, 1))
    Vs = torch.empty(size=(args.N_int, args.n_int, 1))
    Ts = torch.empty(size=(args.N_int, args.n_int, 1))
    Hs = torch.empty(size=(args.N_int, args.n_int, int(args.d_f**2)))
    H_loss_per_traj = []

    if args.model == 'SyMo' or args.model == 'E2E-SyMo':
        x_int, y_int_true, u_int = data_int.get_DEL_data()
        zero_controls = torch.zeros_like(u_int) #add zeros for 2 degree of freedom
        u_int = torch.cat((u_int[:], zero_controls), 2)
        x0_int = x_int[:,0, :int(2*args.d_f)]
        for i in range(args.N_int):
            q_past, q = torch.split(x0_int[i],args.d_f,0)
            y_pred = implicit_integration_DEL(args, nn, x0_int[i], u_int[i], args.int_tol, args.int_maxiter)
            y_int_pred[i] = y_pred
            H_true = cartpole().Inertia_matrix(y_pred.cpu().numpy())
            E = cartpole().energy(y_int_pred[i])
            Energies[i] = E.view(len(E), 1)
            H, V, T = nn.get_matrices(y_pred.to(args.device))
            Vs[i] = V
            Ts[i] = T
            Hs[i] = H.flatten(1,2)
            E_loss_per_traj.append(torch.mean((E_int_true[i] - E)**2))
            H_loss_per_traj.append(np.mean((H_true - Hs[i].cpu().numpy())**2))
            
    elif args.model =='NODE' or args.model == 'LODE':
        x_int, y_int_true, u_int = data_int.get_NODE_data()
        zero_controls = torch.zeros_like(u_int)
        u_int = torch.cat((u_int[:], zero_controls), 2)  #add zeros for 2 degree of freedom
        x0_int = x_int[:,0]
        
        for i in range(args.N_int):
            y_pred = integrate_ODE(args, nn, x0_int[i], u_int[i])
            y_int_pred[i] = y_pred
            E = cartpole().energy(y_int_pred[i])
            E_loss_per_traj.append(torch.mean((E_int_true[i] - E)**2))
            Energies[i] = E.view(len(E), 1)
            
            if args.model == 'LODE':
                H, V, T = nn.get_matrices(y_int_pred[i].to(args.device))
                Vs[i] = V
                Ts[i] = T
                Hs[i] = H.flatten(1,2)
                H_true = cartpole().Inertia_matrix(y_pred.cpu().numpy())
                H_loss_per_traj.append(np.mean((H_true - Hs[i].cpu().numpy())**2))
    
    E_loss_per_traj = torch.stack(E_loss_per_traj)
    if not args.model == 'NODE':
        H_loss_per_traj = np.stack(H_loss_per_traj)
        stats['H_std'] = H_loss_per_traj.std().item()
        stats['H_loss'] = np.mean(H_loss_per_traj).item()
        stats['H_int'] = Hs
        stats['T_int'] = Ts
        stats['V_int'] = Vs
    
    #get integration losses
    criterion = torch.nn.MSELoss(reduction='none')
    losses = criterion(y_int_pred, y_int_true)
    int_loss_per_traj = torch.mean(losses, [1,2])
    int_loss_per_traj_poses = torch.mean(losses, [1])[0]
    stats['int_std'] = int_loss_per_traj.std().item()
    stats['int_loss'] = torch.mean(int_loss_per_traj).detach().cpu().numpy().item()
    
    stats['int_std_poses'] = int_loss_per_traj_poses.std().item()
    stats['int_loss_poses'] = torch.mean(int_loss_per_traj_poses).detach().cpu().numpy().item()
    
    #get energy losses- mean over MSE of each trajectory
    stats['E_std'] = E_loss_per_traj.std().item()
    stats['E_loss'] = torch.mean(E_loss_per_traj).detach().cpu().numpy().item()

    #Save Es for eventual plots
    stats['E_int'] = Energies
    stats['E_int_true'] = E_int_true



    #Save data for reproducibility
    stats['x_train'] = x_train
    stats['y_train'] = y_train
    stats['x_test'] = x_test
    stats['y_test'] = y_test
    stats['x0_int'] = x0_int
    stats['y_int_pred'] = y_int_pred
    stats['y_int_true'] = y_int_true
    
    #just to check losses with reconstructed velocities
    if args.model == 'SyMo' or args.model == 'E2E-SyMo':
        stats['train_loss_vel'] = total_loss(args, model, x_train, y_train)
        stats['test_loss_vel'] = total_loss(args, model, x_test, y_test)
    return nn, stats

def save_model(args, model, stats):
    hyperparameters = vars(args)
    hyperparameters['total_n_params'] = get_n_params(model)
    stats['hyperparameters'] = hyperparameters
    os.makedirs(args.save_dir) if not os.path.exists(args.save_dir) else None
    if args.root_find == 'Broyden' and args.model == 'E2E-SyMo':
        label = '-B-SyMo'
    elif args.root_find == 'Newton' and args.model == 'E2E-SyMo':
        label = '-N-SyMo'
    elif args.model == 'SyMo':
        label = '-SyMo'
    elif args.model == 'LODE':
        label = '-LODE-' + args.odeint 
    elif args.model == 'NODE':
        label = '-NODE-' + args.odeint

    path = '{}/{}{}-p-{}x{}.tar'.format(args.save_dir, args.name, label, args.N_train, args.n_train)
    torch.save(model.state_dict(), path)
    path = '{}/{}{}-p-{}x{}-stats.pkl'.format(args.save_dir, args.name, label, args.N_train, args.n_train)
    to_pickle(stats, path)



if __name__ == "__main__":
    args = get_args()

    n_max_trajs = 128
     #Define the limits for the initial conditions of the generated trajectories
    x0_min = [-4.8, -np.pi, -0.5, -np.pi, -10] #q, q_dot, u_magnitude
    x0_max = [4.8, np.pi, 0.5, np.pi, 10]
    
    #Define the shape of the control wave - random uniform magnitude will be multiplied
    u_train_shape = torch.ones(size=(1, args.n_train+2)) 
    data_train = data(args.name, x0_max, x0_min, args.time_step, n_max_trajs, args.N_test, args.n_train, args.train_seed, u_train_shape, args.noise_std, args.noise_seed)
    
    #Create data for symos
    x_train_symo, y_train_symo, u_train_symo, x_test_symo, y_test_symo, u_test_symo = data_train.get_DEL_data()
    
    #Create data for odes
    x_train_ode, y_train_ode, u_train_ode, x_test_ode, y_test_ode, u_test_ode = data_train.get_NODE_data()
    
    methods = ['LODE', 'NODE']
    n_traj = [16, 32, 64, 128]
    for method in methods:
        args.model = method
        print(method)
        for n in n_traj:
            args.N_train = n
            args.batch_size = int(4*n)
            if method == 'E2E-SyMo':
                root_find = ['Newton', 'Broyden']
                for rf in root_find:
                    args.root_find = rf
                    model, stats = Train(args, x_train_symo[:n], y_train_symo[:n], u_train_symo[:n], x_test_symo, y_test_symo, u_test_symo)
                    save_model(args, model, stats)
            elif method == 'SyMo':
                    model, stats = Train(args, x_train_symo[:n], y_train_symo[:n], u_train_symo[:n], x_test_symo, y_test_symo, u_test_symo)
                    save_model(args, model, stats)
            else:
                    odeint = ['rk4', 'midpoint']
                    for ode in odeint:
                        args.odeint = ode
                        model, stats = Train(args, x_train_ode[:n], y_train_ode[:n], u_train_ode[:n], x_test_ode, y_test_ode, u_test_ode)
                        save_model(args, model, stats)
