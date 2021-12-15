import torch
from Code.root_find import rootfind


def train(method, nn, integrator, criterion, optimizer, scheduler, device, trainloader, testloader, num_epoch, gradient_clipping = None):
    train_losses = []
    train_losses_poses = []
    val_losses = []
    val_single_losses = []
    nn.train()
    n_iterations = []
    norms = []
    integrator.train()
    checkpoint = {}
    for epoch in range(num_epoch):
        running_loss = []
        single_loss = []
        for data in trainloader:
            state, target = data    
            state = state.to(device)
            target = target.to(device)
                
            if (state.dim() or target.dim()) != 2:
                raise Exception('Make sure x_train and y_train are a tensor with 2 dimensions defined by N x n_features/n_outputs, where N is the number of samples of the mini-batch')
            # Clear gradients from the previous iteration
            optimizer.zero_grad()
            
            if method == 'SyMo':
                output = nn(state)
            elif method == "E2E-SyMo":
                output, nstep, diff = integrator(state)
                n_iterations.append(nstep.item())
                norms.append(diff.item())
            else:
                output = integrator(state)
            loss = criterion(output, target) # Calculate the loss

            loss.backward()

            if method == 'SyMo': #calculate mini batched losses - following ML convention
                #get dimensions state = (batch x q_past x q x q_next x u_past x u x u_next)
                integrator.eval()
                d_f = len(state[0])//6
                inp = torch.cat((state[:, :int(2*d_f)], state[:, int(3*d_f):]), 1)
                q_next = integrator(inp).detach()
                loss_single = torch.mean((q_next - target)**2)
                single_loss.append(loss_single.detach().item())
            
            elif method == 'LODE' or method == 'NODE':
                d_f = len(state[0])//3
                q_next = output[:, :d_f].detach()
                loss_single = torch.mean((q_next - target[:, :d_f])**2)
                single_loss.append(loss_single.detach().item())

            else: #E2E SyMos - training loss is the loss for the poses itself
                loss_single = loss
                single_loss.append(loss_single.detach().item())

            running_loss.append(loss.detach().item())
            single_loss.append(loss_single.detach().item())

            if gradient_clipping is not None:
                torch.nn.utils.clip_grad_norm_(integrator.parameters(), gradient_clipping)
            optimizer.step() # Update trainable weights
        
        val_loss, val_single_loss = evaluate(method, nn, integrator, criterion, testloader, device)
    
        #Use the validation loss as a metric for the scheduler
        scheduler.step(val_loss)
        average_loss_epoch = torch.mean(torch.tensor(running_loss)).item()
        average_single_loss_epoch = torch.mean(torch.tensor(single_loss)).item()
        train_losses.append(torch.FloatTensor(running_loss).mean().item())
        val_losses.append(val_loss)
        val_single_losses.append(val_single_loss)
        train_losses_poses.append(average_single_loss_epoch)
        nn.train()
        integrator.train()
        print("Epoch {}: Train_loss:{} Test_loss: {} Lr:{}".format(epoch,average_loss_epoch, val_loss, [g['lr'] for g in optimizer.param_groups])) # Print the average loss for this epoch
        if method == 'SyMo':
            print("rootfind_train_loss:{} rootfind_Test_loss: {}\n".format(average_single_loss_epoch, val_single_loss)) # Print the losses after the external root find
    
    checkpoint['model_state_dict'] = nn.state_dict()
    checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    checkpoint['test_loss'] = val_loss
    checkpoint['train_loss'] = average_loss_epoch
    checkpoint['train_losses'] = train_losses
    checkpoint['test_losses'] = val_losses
    checkpoint['device'] = device
    checkpoint['train_loss_poses'] = average_single_loss_epoch
    checkpoint['train_losses_poses'] = train_losses_poses
    checkpoint['test_loss_poses'] = val_single_loss
    checkpoint['test_losses_poses'] = val_single_losses
    if method == "E2E-SyMo":
        checkpoint['iterations'] = n_iterations
        checkpoint['norms'] = norms
    return checkpoint



def evaluate(method, nn, integrator, criterion, testloader, device): # Evaluate accuracy on validation / test set
    nn.eval() # Set the model to evaluation mode
    integrator.eval()
    val_losses = []
    single_loss = []
    with torch.no_grad(): # Do not calculate gradient to speed up computation
        for data in testloader:
            state, target = data    
            state = state.to(device)
            target = target.to(device)
            if (state.dim() or target.dim()) != 2:
                raise Exception('Make sure x_train and y_train are a tensor with 2 dimensions defined by N x n_features/n_outputs, where N is the number of samples of the mini-batch')
            
            if method == 'SyMo':
                output = nn(state)
            else:
                output = integrator(state)
            
            loss = criterion(output, target)
            val_losses.append(loss.detach().item())

            if method == 'SyMo': #calculate mini batched losses - following ML convention
                #get dimensions state = (batch x q_past x q x q_next x u_past x u x u_next)
                d_f = len(state[0])//6
                inp = torch.cat((state[:, :int(2*d_f)], state[:, int(3*d_f):]), 1)
                q_next = integrator(inp).detach()
                loss_single = torch.mean((q_next - target)**2)
                single_loss.append(loss_single.detach().item())
            
            elif method == 'LODE' or method == 'NODE':
                d_f = len(state[0])//2
                q_next = output[:, :d_f].detach()
                loss_single = torch.mean((q_next - target[:, :d_f])**2)
                single_loss.append(loss_single.detach().item())

            else: #E2E SyMos - training loss is the loss for the poses itself
                loss_single = loss
                single_loss.append(loss_single.detach().item())

    val_loss = torch.mean(torch.tensor(val_losses)).cpu().item()
    average_single_loss = torch.mean(torch.tensor(single_loss)).item()

    return val_loss, average_single_loss
