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
THIS_DIR =  THIS_DIR + "/h=0.01"
from Code.Trainer import train
from Code.symo import SyMo_T
from Code.Rigid_Body import MLP
from Code.Utils import get_n_params, to_pickle, from_pickle
from Code.integrate_models import integrate_ODE, implicit_integration_DEL
from Code.models import pendulum
from Code.Trainer import train
from Code.root_find import rootfind
from Code.NN import LODE_T, NODE_T, ODE
from data import get_dataset, arrange_DEL_dataset, arrange_NODE_dataset
def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--model', type=str, default='E2E-SyMo', help="one of 'NODE', 'L-NODE', 'SyMo' and 'E2E-SyMo'")
    parser.add_argument('--name', default='pendulum', type=str, help='only one option right now')
    parser.add_argument('--num_angles', default=1, type=int, help='number of rotational coordinates')
    parser.add_argument('--embedding', default=True, type=bool, help='If embedding is desired')
    parser.add_argument('--N_train', type=int, default='128', help="Number of different train trajectories")
    parser.add_argument('--n_train', type=int, default='32', help="Number of train observations in each trajectory")
    parser.add_argument('--N_test', type=int, default='128', help="Number of different test trajectories")
    parser.add_argument('--n_test', type=int, default='32', help="Number of test observations in each trajectory")
    parser.add_argument('--N_int', type=int, default='16', help="Number of different test trajectories")
    parser.add_argument('--n_int', type=int, default='500', help="Number of test observations in each trajectory")
    parser.add_argument('--noise_std', type=float, default='0.', help="Induced noise std.")
    parser.add_argument('--train_seed', default=0, type=int, help=' training random seed')
    parser.add_argument('--int_seed', default=1, type=int, help=' integration random seed')
    parser.add_argument('--learning_rate', default=1e-4, type=float, help=' initial learning rate')
    parser.add_argument('--nonlinearity', default='tanh', type=str, help='neural net nonlinearity')
    parser.add_argument('--n_hidden_layers', default=2, type=int, help='number of hidden layers')
    parser.add_argument('--n_neurons', default=128, type=int, help='number of neurons')
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
    parser.add_argument('--forward_maxiter', default=10, type=int, help="If E2E-SyMo specify the maximum number of iterations during forward pass")
    parser.add_argument('--backward_maxiter', default=20, type=int, help="If Broyden specify the maximum number of iterations during backward pass")
    parser.add_argument('--int_tol', default=1e-4, type=float, help="Integration tolerance")
    parser.add_argument('--int_maxiter', default=10, type=int, help="Integration maxiter")
    parser.add_argument('--save_dir', default=THIS_DIR, type=str, help='where to save the trained model')
    parser.add_argument('--gpu', type=int, default=2)
    parser.set_defaults(feature=True)
    return parser.parse_args() 

def total_loss(args, model, x, target):
    model = model.eval()
    q_past, q, u_past, u, u_next =  torch.split(x, args.num_angles, 1)
    q_next = model(x.to(args.device))
    v_next = (q_next - q.to(args.device))/args.time_step
    output = torch.cat((q_next, v_next),1)
    criterion = torch.nn.MSELoss()
    return criterion(output, target.to(args.device))

def get_model(args, nn, model, trainloader, testloader, criterion):
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay) 
    scheduler = ReduceLROnPlateau(optimizer,  'min', patience = args.patience, verbose=True, factor=args.factor)
    return train(args.model, nn, model, criterion, optimizer, scheduler, args.device, trainloader, testloader, args.n_epochs)

def Train(args, x_train, y_train, u_train, x_test, y_test, u_test, data_int, controls_int):

    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
    # reproducibility: set random seed
    args.device = device
    torch.manual_seed(230)
    np.random.seed(230)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    n_angles = args.num_angles
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
    args.d_f = args.num_angles
    
    stats = {}

    #Convert to tensors
    x_train = torch.tensor(x_train)
    y_train = torch.tensor(y_train)
    x_test = torch.tensor(x_test)
    y_test = torch.tensor(y_test)
    u_train = torch.tensor(u_train)
    u_test = torch.tensor(u_test)
    
    #flatten the data
    x_train = torch.flatten(x_train, 0,1)
    y_train = torch.flatten(y_train, 0,1)
    x_test = torch.flatten(x_test, 0,1)
    y_test = torch.flatten(y_test, 0,1)
    u_train = torch.flatten(u_train, 0,1)
    u_test = torch.flatten(u_test, 0,1)

    if (args.model == 'E2E-SyMo') and args.root_find == "Newton":
        args.input_dim = 1
        forward_tol = args.forward_tol
        forward_maxiter = args.forward_maxiter
        nn = SyMo_T(n_angles, n_hidden_layers, n_neurons, h, activation, embedding=embedding, ln=False).to(device)
        model = rootfind(nn, args.root_find, forward_tol, forward_maxiter, analyse=True)

        
        criterion = nn.implicit_loss
        train_data = TensorDataset(x_train, y_train[:, :args.num_angles])
        trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
        test_data = TensorDataset(x_test, y_test[:, :args.num_angles])
        testloader = DataLoader(test_data, batch_size=len(x_test), shuffle=False)
        checkpoint = get_model(args, nn, model, trainloader, testloader, criterion)

    elif (args.model == 'E2E-SyMo') and args.root_find == "Broyden":
        args.input_dim = 1
        forward_tol = args.forward_tol
        forward_maxiter = args.forward_maxiter
        backward_tol = args.backward_tol
        backward_maxiter = args.backward_maxiter
        nn = SyMo_T(n_angles, n_hidden_layers, n_neurons, h, activation, embedding=embedding, ln=False).to(device)
        model = rootfind(nn, args.root_find, forward_tol, forward_maxiter, backward_tol, backward_maxiter)

        
        criterion = nn.implicit_loss
        train_data = TensorDataset(x_train, y_train[:, :args.num_angles])
        trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
        test_data = TensorDataset(x_test, y_test[:, :args.num_angles])
        testloader = DataLoader(test_data, batch_size=len(x_test), shuffle=False)
        checkpoint = get_model(args, nn, model, trainloader, testloader, criterion)

    
    
    elif (args.model == 'SyMo'):
        args.input_dim = 1
        forward_tol = args.forward_tol
        forward_maxiter = args.forward_maxiter
        nn = SyMo_T(n_angles, n_hidden_layers, n_neurons, h, activation, embedding=embedding, ln=False).to(device)
        model = rootfind(nn, args.root_find, forward_tol, forward_maxiter)
        criterion = nn.loss
        d_f = args.d_f

        #add q_next to the input data
        x_train_wq = torch.cat((x_train[:, :int(2*d_f)], y_train[:, 0][:, None], x_train[:, int(2*d_f):]), dim=1)
        x_test_wq = torch.cat((x_test[:, :int(2*d_f)], y_test[:, 0][:, None], x_test[:, int(2*d_f):]), dim=1)
        
        train_data = TensorDataset(x_train_wq, y_train[:, :args.num_angles])
        trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
        test_data = TensorDataset(x_test_wq, y_test[:, :args.num_angles])
        testloader = DataLoader(test_data, batch_size=len(x_test), shuffle=False)
        checkpoint = get_model(args, nn, model, trainloader, testloader, criterion)




    elif (args.model == 'L-NODE'):
        args.input_dim = 1
        nn = LODE_T(n_angles, n_hidden_layers, n_neurons, activation, embedding=embedding, ln=False).to(device)
        model = ODE(nn, args.odeint, h).to(device)
        criterion = nn.loss
        x_train = torch.cat((x_train, u_train), dim=1)
        train_data = TensorDataset(x_train, y_train)
        trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
        x_test = torch.cat((x_test, u_test), dim=1)
        test_data = TensorDataset(x_test, y_test)
        testloader = DataLoader(test_data, batch_size=len(x_test), shuffle=False)
        checkpoint = get_model(args, nn, model, trainloader, testloader, criterion)
        
    
    elif (args.model == 'NODE'):
        args.input_dim = 3
        u_index = 0
        nn = NODE_T(n_angles, n_hidden_layers, n_neurons, activation, u_index, embedding=embedding, ln=False).to(device)
        model = ODE(nn, args.odeint, h).to(device)
        criterion = nn.loss
        x_train = torch.cat((x_train, u_train), dim=1)
        train_data = TensorDataset(x_train, y_train)
        trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
        x_test = torch.cat((x_test, u_test), dim=1)
        test_data = TensorDataset(x_test, y_test)
        testloader = DataLoader(test_data, batch_size=len(x_test), shuffle=False)
        checkpoint = get_model(args, nn, model, trainloader, testloader, criterion)
        
    else:
        raise ValueError('model_type not supported')
    

    stats = checkpoint.copy()

    #Integrate, compute energies, masses and respective losses 
    H_true = pendulum().H
    y_int_pred = torch.zeros(size=(args.N_int, args.n_int, int(2*args.d_f)))
    E_loss_per_traj = []
    Energies = torch.empty(size=(args.N_int, args.n_int, 1))
    Vs = torch.empty(size=(args.N_int, args.n_int, 1))
    Ts = torch.empty(size=(args.N_int, args.n_int, 1))
    Hs = torch.empty(size=(args.N_int, args.n_int, int(args.d_f**2)))
    H_loss_per_traj = []

    if args.model == 'SyMo' or args.model == 'E2E-SyMo':
        x_int, y_int_true, _ = arrange_DEL_dataset(data_int['test_x'], controls_int['test_u'])
        x_int = torch.tensor(x_int)
        y_int_true = torch.tensor(y_int_true)
        E_int_true = pendulum().energy(data_int['x0'])

        x0_int = x_int[:,0, :2]
        for i in range(args.N_int):
            y_pred = implicit_integration_DEL(args.root_find, args.n_int, args.time_step, nn.eval(), x0_int[i], args.int_tol, args.int_maxiter, device)
            y_int_pred[i] = y_pred.cpu()
            E = pendulum().energy(y_int_pred[i])
            Energies[i] = E.view(len(E), 1)
            H, V, T = nn.get_matrices(y_pred.to(args.device))
            Vs[i] = V
            Ts[i] = T
            Hs[i] = H.flatten(1,2)
            E_loss_per_traj.append(torch.mean((E_int_true[i] - E)**2))
            H_loss_per_traj.append(torch.mean((H_true - H)**2))
            
    elif args.model =='NODE' or args.model == 'L-NODE':
        x_int, y_int_true, _ = arrange_NODE_dataset(data_int['test_x'], controls_int['test_u'])
        x_int = torch.tensor(x_int)
        y_int_true = torch.tensor(y_int_true)
        E_int_true = pendulum().energy(data_int['x0'])
        x0_int = x_int[:,0]
        
        for i in range(args.N_int):
            y_pred = integrate_ODE(nn.eval(), args.odeint, x0_int[i], args.n_int, args.time_step, args.device)
            y_int_pred[i] = y_pred.cpu()
            E = pendulum().energy(y_int_pred[i])
            E_loss_per_traj.append(torch.mean((E_int_true[i] - E)**2))
            Energies[i] = E.view(len(E), 1)
            
            if args.model == 'L-NODE':
                H, V, T = nn.get_matrices(y_int_pred[i].to(args.device))
                Vs[i] = V
                Ts[i] = T
                Hs[i] = H.flatten(1,2)
                H_loss_per_traj.append(torch.mean((H_true - H)**2))
    
    E_loss_per_traj = torch.stack(E_loss_per_traj)
    if not args.model == 'NODE':
        H_loss_per_traj = torch.stack(H_loss_per_traj)
        stats['H_std'] = H_loss_per_traj.std().item()
        stats['H_loss'] = torch.mean(H_loss_per_traj).detach().cpu().numpy().item()
        stats['H_int'] = Hs.numpy()
        stats['T_int'] = Ts.numpy()
        stats['V_int'] = Vs.numpy()
    
    #get integration losses
    criterion = torch.nn.MSELoss(reduction='none')
    losses = criterion(y_int_pred, y_int_true)
    int_loss_per_traj = torch.mean(losses, [1,2])
    int_loss_per_traj_poses = torch.mean(losses, [1])[:, 0]
    stats['int_std'] = int_loss_per_traj.std().item()
    stats['int_loss'] = torch.mean(int_loss_per_traj).detach().cpu().numpy().item()
    
    stats['int_std_poses'] = int_loss_per_traj_poses.std().item()
    stats['int_loss_poses'] = torch.mean(int_loss_per_traj_poses).detach().cpu().numpy().item()
    
    #get energy losses- mean over MSE of each trajectory
    stats['E_std'] = E_loss_per_traj.std().item()
    stats['E_loss'] = torch.mean(E_loss_per_traj).detach().cpu().numpy().item()
    #Save Es for eventual plots
    stats['E_int'] = Energies.numpy()
    stats['E_int_true'] = E_int_true


    #Save data for reproducibility
    stats['int_losses'] = losses.numpy()
    stats['x_train'] = x_train.numpy()
    stats['y_train'] = y_train.numpy()
    stats['x_test'] = x_test.numpy()
    stats['y_test'] = y_test.numpy()
    stats['x0_int'] = x0_int.numpy()
    stats['y_int_pred'] = y_int_pred.numpy()
    stats['y_int_true'] = y_int_true.numpy()
    
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
    elif args.model == 'L-NODE':
        label = '-L-NODE-' + args.odeint 
    elif args.model == 'NODE':
        label = '-NODE-' + args.odeint

    path = '{}/{}{}-p-{}x{}.tar'.format(args.save_dir, args.name, label, args.N_train, args.n_train)
    torch.save(model.state_dict(), path)
    path = '{}/{}{}-p-{}x{}-stats.pkl'.format(args.save_dir, args.name, label, args.N_train, args.n_train)
    to_pickle(stats, path)



if __name__ == "__main__":
    args = get_args()

    n_max_trajs = 128
    args.N_train = n_max_trajs
    args.N_test = n_max_trajs

    #train with constant inputs
    us = [-2, 2]
    data, controls = get_dataset(seed=args.train_seed, time_steps=args.n_train, h=args.time_step, us=us, u_shape='constant', samples=int(2*n_max_trajs))
    full_data = {"data" : data, "controls" : controls}
    
    #int data
    data_int, controls_int = get_dataset(seed=args.int_seed, time_steps=args.n_int, h=args.time_step, us=us, u_shape='zeros', samples=args.N_int, test_split=0)
    full_data['int_data'] = {"data": data_int, "controls": controls_int}

    #save data for reproducibility
    path = '{}/{}.pkl'.format(args.save_dir, "dataset")
    to_pickle(full_data, path)
    
    #Create data for symos
    x_train_symo, y_train_symo, u_train_symo = arrange_DEL_dataset(data['train_x'], controls['train_u'])
    x_test_symo, y_test_symo, u_test_symo = arrange_DEL_dataset(data['test_x'], controls['test_u'])
    
    #Create data for odes
    x_train_ode, y_train_ode, u_train_ode = arrange_NODE_dataset(data['train_x'], controls['train_u'])
    x_test_ode, y_test_ode, u_test_ode = arrange_NODE_dataset(data['test_x'], controls['test_u'])
    
    methods = ['E2E-SyMo', 'SyMo', 'L-NODE','NODE']
    n_traj = [8, 16, 32,64, 128]
    for method in methods:
        args.model = method
        print(method)
        for n in n_traj:
            args.N_train = n
            args.batch_size = int(n*4)
            if method == 'E2E-SyMo':
                root_find = ['Newton']
                for rf in root_find:
                    args.root_find = rf
                    model, stats = Train(args, x_train_symo[:n], y_train_symo[:n], u_train_symo[:n], x_test_symo, y_test_symo, u_test_symo, data_int, controls_int)
                    save_model(args, model, stats)
            elif method == 'SyMo':
                    model, stats = Train(args, x_train_symo[:n], y_train_symo[:n], u_train_symo[:n], x_test_symo, y_test_symo, u_test_symo, data_int, controls_int)
                    save_model(args, model, stats)
            else:
                    odeint = ['midpoint', 'rk4']
                    for ode in odeint:
                        args.odeint = ode
                        model, stats = Train(args, x_train_ode[:n], y_train_ode[:n], u_train_ode[:n], x_test_ode, y_test_ode, u_test_ode, data_int, controls_int)
                        save_model(args, model, stats)
