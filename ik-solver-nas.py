import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import math
import torch
import time
import os
import csv
import torch.utils.data
import torch.nn as nn
import pandas as pd
import random
import torch.optim as optim
#import robust_loss_pytorch
import datetime
import sys
#import tools
from utils import *
import optuna
from optuna.trial import TrialState
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
#from torch.utils.tensorboard import SummaryWrite
#from torchvision import models
#from torchsummary import summary


# We optimize:
"""
1. Number of hidden layers                      : [1, 20]
2. Number of hidden neurons / hidden layers     : [128, 1024]
3. Optimizer type                               : ["Adam", "RMSprop", "SGD"]
4. Learning rates                               : [1e-5, 1e-1]
5. Activation functions                         : ["sigmoid", "relu", "tanh", "lrelu"]
"""

# Global variables
DEVICE = torch.device("cuda:0")
MODEL = "MLP"
EPOCHS = 1000
NUM_DOF = 7
INPUT_DIM = 6
ROBOT_CHOICE = "7DoF-7R-Panda" # "7DoF-7R-Panda", "7DoF-GP66"
#PATH = "/home/retina/dembysj/Documents/"+ROBOT_CHOICE+"_optuna_search"
PATH = ROBOT_CHOICE+"_optuna_search_final"
PRINTING_STEP = 10
DATASET_SAMPLES = 1000000
SCALE = 10 
BATCH_SIZE = 100000  
DATASET_TYPE = "1_to_1"


# Define the network search space
def define_model(trial):
    
    # 1. Optimize number of layers:
    n_layers = trial.suggest_int("n_layers", 1, 15)
    layers = []
    
    in_features = INPUT_DIM # number of joints
    
    activations = {
        'relu': nn.ReLU(),
        'lrelu': nn.LeakyReLU(),
        'sigmoid': nn.Sigmoid(),
        'tanh': nn.Tanh()
    }
    
    for i in range(n_layers):
        #out_features = trial.suggest_int("n_units_1{}".format(i), 128, 1024)
        out_features = trial.suggest_int("n_units_1{}".format(i), 1024, 1024)
        
        layers.append(nn.Linear(in_features, out_features))
        
        #activation_name = trial.suggest_categorical("activation", ["sigmoid", "relu", "tanh", "lrelu",])
        activation_name = trial.suggest_categorical("activation", ["relu"])
        activation = activations[activation_name]
        layers.append(activation)

        #layers.append(nn.Sigmoid())
        in_features = out_features
    
    layers.append(nn.Linear(in_features, NUM_DOF))
    
    return nn.Sequential(*layers)


def define_model_rmlp(trial):
    
    # 1. Optimize number of layers:
    n_layers = trial.suggest_int("n_layers", 1, 20)
    layers = []
    
    in_features = INPUT_DIM # number of joints
    
    activations = {
        'relu': nn.ReLU(),
        'lrelu': nn.LeakyReLU(),
        'sigmoid': nn.Sigmoid(),
        'tanh': nn.Tanh()
    }
    
    for i in range(n_layers):
        #out_features = trial.suggest_int("n_units_1{}".format(i), 128, 1024)
        out_features = trial.suggest_int("n_units_1{}".format(i), 1024, 1024)
        
        layers.append(nn.Linear(in_features, out_features))
        
        #activation_name = trial.suggest_categorical("activation", ["sigmoid", "relu", "tanh", "lrelu",])
        activation_name = trial.suggest_categorical("activation", ["relu"])
        activation = activations[activation_name]
        layers.append(activation)

        #layers.append(nn.Sigmoid())
        in_features = out_features
    
    layers.append(nn.Linear(in_features, NUM_DOF))
    
    return nn.Sequential(*layers)



# Define the objective function
def objective(trial):
    
    # Generate the model
    model = define_model(trial).to(DEVICE)
    
    # Generate the optimizers.
    #lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    lr = trial.suggest_float("lr", 1e-3, 1e-3, log=True)
    #optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])    
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam"])    
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
    
    #optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_function = torch.nn.MSELoss()
    
    # Get the dataset.
    dataset = pd.read_csv('../docker/datasets/'+ROBOT_CHOICE+'/data_'+ROBOT_CHOICE+'_'+str(int(DATASET_SAMPLES))+'_qlim_scale_'+str(int(SCALE))+'_seq.csv')
    dataset_a = np.array(dataset)
    #train_loader, valid_loader = get_data()
    train_loader, valid_loader, X_validate, y_validate, X_train, y_train, X_test, y_test = load_dataset(dataset_a, 
                                                                                                        NUM_DOF, 
                                                                                                        BATCH_SIZE, 
                                                                                                        ROBOT_CHOICE, 
                                                                                                        DATASET_TYPE, 
                                                                                                        DEVICE, 
                                                                                                        INPUT_DIM)
    #print(trial.params)
    #sys.exit()
    
    min_valid_loss = np.inf
    best_epoch = 0
    loss_history = []
    
    # Training the model
    start_time = time.monotonic()
    for epoch in range(EPOCHS):
        #print(EPOCHS)
        #print("\nCurrent Study epoch: {}\n".format(epoch))
        
        running_loss = 0.0
        model.train()
        for (batch_idx, batch) in enumerate(train_loader):
            optimizer.zero_grad()
            predicted_output = model(batch["input"].to(DEVICE))
            loss = loss_function(predicted_output, batch["output"].to(DEVICE))
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        epoch_loss = running_loss/len(train_loader)
            
        # validation of the model
        running_loss_v = 0.0
        model.eval()
        with torch.no_grad():
            for (batch_idx_v, batch_v) in enumerate(valid_loader):
                
                predicted_output = model(batch_v["input"].to(DEVICE))
                loss_v = loss_function(predicted_output, 
                                       batch_v["output"].to(DEVICE))
                
                running_loss_v += loss_v.item()
                
            epoch_loss_v = running_loss_v/len(valid_loader)

        loss_history.append([epoch_loss, epoch_loss_v])
            
        if (epoch % PRINTING_STEP) == 0:
            end_time = time.monotonic()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)            
            print('\nEpoch: {}/{} | Epoch Time: {}m {}s'.format(epoch, EPOCHS, epoch_mins, epoch_secs))
            print('\tTrain Loss: {}'.format(epoch_loss))
            print('\tValid Loss: {}'.format(epoch_loss_v))
            print("\tBest Epoch Occurred [{}/{}]".format(best_epoch, EPOCHS)) 
                

            if epoch_loss_v < min_valid_loss:
                #print("valid loss decreased: {:,.6f} ---> {:,.6f}\n".format(min_valid_loss, epoch_loss_v))
                min_valid_loss = epoch_loss_v
                best_epoch = epoch
                        

                # saving the model at this checkpoint (save best validation results)                                
                torch.save(model.state_dict(), PATH+'/saved_model_'+ROBOT_CHOICE+'_'+str(epoch)+'_'+optimizer_name+'.pth')       
                path_save_model = PATH + '/model_'+ROBOT_CHOICE+'_'+str(epoch)+'_'+optimizer_name+'.pt'
                torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
                }, path_save_model)
            
        trial.report(epoch_loss_v, epoch)
        
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    
    header = ["training_loss", "validation_loss"]
    df = pd.DataFrame(loss_history)
    df.to_csv(PATH+"/loss_history_"+ROBOT_CHOICE+"_"+str(DATASET_SAMPLES)+"_nlayers_"+ str(trial.params["n_layers"]) +".csv",
        index=False,
        header=header)
        
    return epoch_loss_v

if __name__ == "__main__":    
    
    
    is_exist = os.path.exists(PATH)
    if not is_exist:
        os.makedirs(PATH)
        
    sampler = optuna.samplers.TPESampler(seed=10)
    study = optuna.create_study(study_name=ROBOT_CHOICE+"_kinematics_search_seq_"+MODEL,
                                storage='sqlite:///'+ROBOT_CHOICE+'_kinematics_search_seq_'+MODEL+'.db',
                                direction="minimize",
                                sampler=sampler)
    #study.optimize(objective, n_trials=100, timeout=600)
    study.optimize(objective, n_trials=20, timeout=None)
    
    df = study.trials_dataframe()
    df.to_csv(PATH+'/optuna_history.csv', index=False)
     
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
     
     
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
        
    fig_hist = optuna.visualization.plot_optimization_history(study)
    fig_hist.write_html(PATH+"/optuna_history.html")
    #fig.show()
    
    fig_importance = optuna.visualization.plot_param_importances(study)
    fig_importance.write_html(PATH+"/optuna_importance.html")
    
    fig_edf = optuna.visualization.plot_edf([study])
    fig_edf.write_html(PATH+"/optuna_edf.html")
    
    
    fig_inter = optuna.visualization.plot_intermediate_values(study)
    fig_inter.write_html(PATH+"/optuna_inter.html")
    
    fig_relation = optuna.visualization.plot_parallel_coordinate(study)
    fig_relation.write_html(PATH+"/optuna_relation.html")
    
    #fig_pareto = optuna.visualization.plot_pareto_front(study)
    #fig_pareto.write_html(PATH+"/optuna_pareto.html")
    
    fig_slice = optuna.visualization.plot_slice(study)
    fig_slice.write_html(PATH+"/optuna_slice.html")
    
    
    #optuna.visualization.matplotlib.plot_optimization_history(study)
     
