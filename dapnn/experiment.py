'''
Created on Jun 28, 2018

@author: david
'''

import dapnn.nn as nn
import torch
import numpy as np


graph_colors = {0 : '#0000FF', 1 : '#FF0000', 2 : '#00FF00', 3 : '#00FFFA', 4 : '#A100FF', 5 : '#FAFF00', 6 : '#000000', 7 : '#FF00F6', 8 : '#00FFB2', 9 : '#FF006E'}

class experiment(object):

    def __init__(self, models, params, names, train_dataloader, val_dataloader, exp_nbr=0 ,startseed=0, modelcnt=5):
        assert len(models) == len(params)
        assert len(models) == len(names)
        self.models = models
        self.params = params
        self.names = names
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.startseed = startseed
        self.modelcnt = modelcnt
        self.exp_nbr = exp_nbr
        
        
    def run(self, timepermodel=60, restart=True):
        
        for i in range(0, self.modelcnt):
            for m in range(0, len(self.models)):
                
                self.params[m]['seed'] = self.startseed + i
                model = self.models[m](**self.params[m], name='{}-{}-{}'.format(self.names[m], self.exp_nbr, self.params[m]['seed']))
                
                if restart is False:
                    print('loading model: ' + model.name)
                    model.load_state_dict(torch.load('./models/current_' + model.name + '.ckpt'))
                    
                nn.train_model(model, self.train_dataloader, self.val_dataloader, train_sec=timepermodel)
                del model
                
                
    def graph_results(self):
        
        train_mins = np.zeros([len(self.models), self.modelcnt])
        train_finals = np.zeros([len(self.models), self.modelcnt])
        
        val_mins = np.zeros([len(self.models), self.modelcnt])
        val_finals = np.zeros([len(self.models), self.modelcnt])
        
        
        for i in range(0, self.modelcnt):
            for m in range(0, len(self.models)):
                
                self.params[m]['seed'] = self.startseed + i
                model = self.models[m](**self.params[m], name='{}-{}-{}'.format(self.names[m], self.exp_nbr, self.params[m]['seed']))
                
                #print('loading model: ' + model.name)
                model.load_state_dict(torch.load('./models/current_' + model.name + '.ckpt'))
                
                plot.add_plot_line(model.train_batch_loss_x.numpy(), model.train_batch_loss_y.numpy(), fmt=graph_colors[m],
                       datalabel='{}'.format(self.names[m]), xlabel='BatchNbr', ylabel='Loss', title='Batch Train Loss', figure=1)
                
                plot.add_plot_line(np.convolve(model.train_batch_loss_x.numpy(), 10 * [0.1], mode='valid'), np.convolve(model.train_batch_loss_y.numpy(), 10 * [0.1], mode='valid'), fmt=graph_colors[m],
                       datalabel='{}'.format(self.names[m]), xlabel='BatchNbr', ylabel='Loss', title='Avg Batch Train Loss', figure=2)
                
                plot.add_plot_line(model.checkpoint_x.numpy(), model.val_ckpt_loss_y.numpy(), fmt=graph_colors[m],
                       datalabel='{}'.format(self.names[m]), xlabel='BatchNbr', ylabel='Loss', title='Ckpt Val Loss', figure=3)
                
                avg_train_batch_loss_y = np.convolve(model.train_batch_loss_y.numpy(), 10 * [0.1], mode='valid')
                
                train_mins[m][i] = avg_train_batch_loss_y.min()
                train_finals[m][i] = avg_train_batch_loss_y[-1]
                
                val_mins[m][i] = model.val_ckpt_loss_y.numpy().min()
                val_finals[m][i] = model.val_ckpt_loss_y.numpy()[-1]
                
                
                del model
        
        train_mins_mean, train_mins_std = train_mins.mean(axis=1), train_mins.std(axis=1)
        train_finals_mean, train_finals_std = train_finals.mean(axis=1), train_finals.std(axis=1)
        val_mins_mean, val_mins_std = val_mins.mean(axis=1), val_mins.std(axis=1)
        val_finals_mean, val_finals_std = val_finals.mean(axis=1), val_finals.std(axis=1)

        
        for i in range(0, len(self.models)):
            print('Model {}: Train Min: {:.4f}+/-{:.4f} , Train Final: {:.4f}+/-{:.4f} , Val Min: {:.4f}+/-{:.4f} , Val Final: {:.4f}+/-{:.4f}'.format('{}'.format(self.names[i]), train_mins_mean[i], train_mins_std[i], train_finals_mean[i], train_finals_std[i], 
                                                                                                 val_mins_mean[i], val_mins_std[i], val_finals_mean[i], val_finals_std[i]))
        
        plot.show()
        
        
    def model_losses(self, dataloader):
        
        best = np.zeros([len(self.models), self.modelcnt])
        current = np.zeros([len(self.models), self.modelcnt])       
                
        for i in range(0, self.modelcnt):
            for m in range(0, len(self.models)):
                
                self.params[m]['seed'] = self.startseed + i
                model = self.models[m](**self.params[m], name='{}-{}-{}'.format(self.names[m], self.exp_nbr, self.params[m]['seed']))
                
                #print('loading model: ' + model.name)
                model.load_state_dict(torch.load('./models/best_' + model.name + '.ckpt'))
                
                best[m][i] = nn.model_loss(model, dataloader)
                
                #print('loading model: ' + model.name)
                model.load_state_dict(torch.load('./models/current_' + model.name + '.ckpt'))
                
                current[m][i] = nn.model_loss(model, dataloader)
                
                del model
                
        best_mean, best_std = best.mean(axis=1), best.std(axis=1)
        current_mean, current_std = current.mean(axis=1), current.std(axis=1)   
        
        for i in range(0, len(self.models)):
            print('Model {}: Best: {:.4f}+/-{:.4f} , Current: {:.4f}+/-{:.4f} '.format('{}'.format(self.names[i]), best_mean[i], best_std[i], current_mean[i], current_std[i]))
        
        return best_mean, best_std, current_mean, current_std
    
    
    def model_accuracies(self, dataloader):
        
        best = np.zeros([len(self.models), self.modelcnt])
        current = np.zeros([len(self.models), self.modelcnt])       
                
        for i in range(0, self.modelcnt):
            for m in range(0, len(self.models)):
                
                self.params[m]['seed'] = self.startseed + i
                model = self.models[m](**self.params[m], name='{}-{}-{}'.format(self.names[m], self.exp_nbr, self.params[m]['seed']))
                
                #print('loading model: ' + model.name)
                model.load_state_dict(torch.load('./models/best_' + model.name + '.ckpt'))
                
                best[m][i] = nn.classifier_accuracy(model, dataloader)
                
                #print('loading model: ' + model.name)
                model.load_state_dict(torch.load('./models/current_' + model.name + '.ckpt'))
                
                current[m][i] = nn.classifier_accuracy(model, dataloader)
                
                del model
                
        best_mean, best_std = best.mean(axis=1), best.std(axis=1)
        current_mean, current_std = current.mean(axis=1), current.std(axis=1)   
        
        for i in range(0, len(self.models)):
            print('Model {}: Best: {:.4f}+/-{:.4f} , Current: {:.4f}+/-{:.4f} '.format('{}'.format(self.names[i]), best_mean[i], best_std[i], current_mean[i], current_std[i]))
        
        return best_mean, best_std, current_mean, current_std
    
    
    
    def get_best_model(self, dataloader):
        
        best_model_loss = float('inf')
        best_model = None
        
        for i in range(0, self.modelcnt):
            for m in range(0, len(self.models)):
                
                self.params[m]['seed'] = self.startseed + i
                model = self.models[m](**self.params[m], name='{}-{}-{}'.format(self.names[m], self.exp_nbr, self.params[m]['seed']))
                
                model.load_state_dict(torch.load('./models/best_' + model.name + '.ckpt'))
                
                loss= nn.model_loss(model, dataloader)
                
                if best_model_loss > loss:
                    best_model_loss = loss
                    best_model = model
                
        return best_model

    def get_best_models(self, dataloader):
        
        best_models = list()
        
        for m in range(0, len(self.models)):
            
            best_model_loss = float('inf')
            best_model = None
            
            for i in range(0, self.modelcnt):
                
                self.params[m]['seed'] = self.startseed + i
                model = self.models[m](**self.params[m], name='{}-{}-{}'.format(self.names[m], self.exp_nbr, self.params[m]['seed']))
                
                model.load_state_dict(torch.load('./models/best_' + model.name + '.ckpt'))
                
                loss= nn.model_loss(model, dataloader)
                
                if best_model_loss > loss:
                    best_model_loss = loss
                    best_model = model
                
            best_models.append(best_model)
            
        return best_models
