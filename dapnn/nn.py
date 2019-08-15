'''
Created on Jun 27, 2018

@author: david
'''

import torch
import torch.cuda
import time
import numpy as np

# Device configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def classifier_accuracy(model, dataloader):
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        
        correct = 0.
        total = 0.
    
        for _, batch in enumerate(dataloader):  
            # Move tensors to the configured device
            batch = list(map(lambda x: x.to(device), batch))

            # Forward pass
            c, t = model.accuracy(batch)
            correct += c
            total += t

    model.train()
    return correct / total


# return loss, lower is better
def model_loss(model, dataloader):
    
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        
        total_loss = 0
        count = 0
    
        for _, batch in enumerate(dataloader):  
            # Move tensors to the configured device
            batch = list(map(lambda x: x.to(device), batch))
            
            # Forward pass
            total_loss += model.loss(batch)
            count += 1
            
    model.train()
    return total_loss / count
    
def do_checkpoint(model, running_train_loss, val_dataloader, cur_time, epoch_nbr, batch_nbr, best_val_loss):
    #train_loss = model_loss(model, train_dataloader)
    val_loss = model_loss(model, val_dataloader)
    print ('Time: {:.4f}, Epoch: {}, Batch: {}, Running Train Loss: {:.4f}, Val Loss: {:.4f}'.format(cur_time, epoch_nbr, batch_nbr, running_train_loss, val_loss.item()))
    
    #model.train_ckpt_loss_y = torch.cat([model.train_ckpt_loss_y, torch.tensor([train_loss]).to(device)])    
    model.val_ckpt_loss_y = torch.cat([model.val_ckpt_loss_y, torch.tensor([val_loss]).to(device)])
    model.checkpoint_x = torch.cat([model.checkpoint_x, torch.tensor([batch_nbr]).to(device)])
    
    
    if(best_val_loss > val_loss):
        print('saving model: ' + model.name)
        torch.save(model.state_dict(), './models/best_' + model.name + '.ckpt')
        best_val_loss = val_loss
    
    return best_val_loss
    
def train_model(model, train_dataloader, val_dataloader, train_sec=60):
    print('training: {} parameter count: {}'.format(model.name, model.count_parameters()))
    
    model = model.to(device)
    model.train()
    
    best_val_loss = model_loss(model, val_dataloader)
    print('val_loss: {:.4f}'.format(best_val_loss.item()))
    
    epoch_nbr = 0
    batch_nbr = 0
    
    start = time.time()
    num_checkpoints = 0;
    
    running_train_loss = best_val_loss.item()
    
    while (time.time() - start) < train_sec:
        epoch_nbr += 1
        for _, batch in enumerate(train_dataloader): 
            batch_nbr += 1 
            # Move tensors to the configured device
            batch = list(map(lambda x: x.to(device), batch))
            
            # Forward pass
            loss = model.loss(batch)
            running_train_loss = running_train_loss * 0.9 + loss.item() * 0.1
            
            model.train_batch_loss_y = torch.cat([model.train_batch_loss_y, torch.tensor([loss]).to(device)])
            model.train_batch_loss_x = torch.cat([model.train_batch_loss_x, torch.tensor([batch_nbr]).to(device)])
            
            # Backward and optimize
            model.optimizer.zero_grad()
            loss.backward()
            model.optimizer.step()
            
            if (time.time() - start) > 20 * num_checkpoints + 1:
                best_val_loss = do_checkpoint(model, running_train_loss, val_dataloader, (time.time() - start), epoch_nbr, batch_nbr, best_val_loss)
                num_checkpoints += 1
                
            if (time.time() - start) > train_sec:
                break    
    
    do_checkpoint(model, running_train_loss, val_dataloader, (time.time() - start), epoch_nbr, batch_nbr, best_val_loss)
    
    print('saving model: ' + model.name)
    torch.save(model.state_dict(), './models/current_' + model.name + '.ckpt')
    
    print('loading model: ' + model.name)
    model.load_state_dict(torch.load('./models/best_' + model.name + '.ckpt'))
    
    return
