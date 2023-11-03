# Libraries
# import libraries
#from mpl_toolkits.mplot3d import Axes3D

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import pandas as pd
import random
import sklearn
import time
import math
#import matplotlib.pyplot as plt
#import os
import sys
import wandb
import yaml

from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import decomposition
from sklearn import manifold
from tqdm import tqdm
from scipy import stats
#from torchviz import make_dot
from utils import *


if __name__ == '__main__':

    
    print('==> Reading from the config file ...')

    # Read parameters from configuration file
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path",
                        type=str,
                        default="./configs/train.yaml",
                        help="Path to train config file.")
    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        config = yaml.full_load(f)

    #print(config)
    
    # set parameters and configurations
    robot_choice = config["ROBOT_CHOICE"]
    seed_choice = config["SEED_CHOICE"]                                           # seed random generators for reproducibility
    seed_number = config["SEED_NUMBER"]
    print_epoch = config["TRAIN"]["PRINT_EPOCHS"]  
    batch_size = config["TRAIN"]["HYPERPARAMETERS"]["BATCH_SIZE"]                 # desired batch size
    init_type = config["TRAIN"]["HYPERPARAMETERS"]["WEIGHT_INITIALIZATION"]       # weights init method (default, uniform, normal, xavier_uniform, xavier_normal)
    #hidden_layer_sizes = [128,128,128,128]                                       # architecture to employ
    learning_rate = config["TRAIN"]["HYPERPARAMETERS"]["LEARNING_RATE"]           # learning rate
    optimizer_choice = config["TRAIN"]["HYPERPARAMETERS"]["OPTIMIZER_NAME"]       # optimizers (SGD, Adam, Adadelta, RMSprop)
    loss_choice =  config["TRAIN"]["HYPERPARAMETERS"]["LOSS"]                     # l2, l1, lfk
    network_type =  config["MODEL"]["NAME"]     
    dataset_samples = config["TRAIN"]["DATASET"]["NUM_SAMPLES"]                   # MLP, ResMLP, DenseMLP, FouierMLP 
    scale = config["TRAIN"]["DATASET"]["JOINT_LIMIT_SCALE"]
    print_steps = config["TRAIN"]["PRINT_STEPS"] 
    save_option = config["TRAIN"]["CHECKPOINT"]["SAVE_OPTIONS"]                                # local or cloud
    EPOCHS = config["TRAIN"]["HYPERPARAMETERS"]["EPOCHS"]                         # total training epochs

    

    
    print('==> Log into wandb to send out metrids ...')
    wandb.login()                                        # login to the Weights and Biases    

        
    
    print("==> Running based on configuration...")
    #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')         # device to train on
    device = torch.device('cuda:'+str(config["DEVICE_ID"])) 
    #device = torch.device('cpu')
    
    # set input and output size based on robot
    if robot_choice == "6DoF-6R-Puma260":
        n_DoF = 6
        input_dim = 6
        output_dim = 6
        pose_header = ["x", "y", "z","R","P","Y"]
        joint_header = ["t1", "t2", "t3", "t4", "t5", "t6"]
    if robot_choice == "7DoF-7R-Panda":
        n_DoF = 7
        input_dim = 6
        output_dim = 7
        pose_header = ["x", "y", "z","R","P","Y"]
        joint_header = ["t1", "t2", "t3", "t4", "t5", "t6", "t7"]
        
    # load dataset from file
    if save_option == "cloud":
        data = pd.read_csv('/home/datasets/'+robot_choice+'/data_'+robot_choice+'_'+str(int(dataset_samples))+'_qlim_scale_'+str(int(scale))+'.csv')
    elif save_option == "local":
        data = pd.read_csv('../docker/datasets/'+robot_choice+'/data_'+robot_choice+'_'+str(int(dataset_samples))+'_qlim_scale_'+str(int(scale))+'.csv')
    data_a = np.array(data) 




    # number of times the experiments needs to be repeated 
    experiments = config["NUM_EXPERIMENT_REPETITIONS"]
    layers = config["MODEL"]["NUM_HIDDEN_LAYERS"]
    neurons = config["MODEL"]["NUM_HIDDEN_NEURONS"]   


    # set the hidden layer array to initialize the architecture
    hidden_layer_sizes = np.zeros((1,layers))          
    
    #for neuron in range(200, neurons+100, 100):


    hidden_layer_sizes[:,:] = neurons
    hidden_layer_sizes = hidden_layer_sizes.squeeze(0).astype(int).tolist()

    for experiment_number in range(experiments):
    
        # ensure reproducibilities if seed is set to true
        if seed_choice:   
            random.seed(seed_number)
            np.random.seed(seed_number)
            torch.manual_seed(seed_number)
            torch.cuda.manual_seed(seed_number)
            torch.backends.cudnn.deterministic = True
        ## train and validate
        # load the dataset
        train_data_loader, test_data_loader, X_validate, y_validate, X_train, y_train, X_test, y_test = load_dataset(data_a, n_DoF, batch_size, robot_choice)

        #print(input_dim)
        #print(hidden_layer_sizes)
        #print(output_dim)

        
        # get network architecture
        if network_type == "MLP":
            model = MLP(input_dim, hidden_layer_sizes, output_dim)
            #model = MLP(mapping_size*2, hidden_layer_sizes, output_dim)
        elif network_type == "ResMLP":
            model = ResMLP(input_dim, hidden_layer_sizes, output_dim)
        elif network_type == "DenseMLP":
            model = DenseMLP(input_dim, hidden_layer_sizes, output_dim)
        elif network_type == "FourierMLP":
            fourier_dim = 16
            scale = 10
            model = FourierMLP(input_dim, fourier_dim, hidden_layer_sizes, output_dim, scale)
            
        
        if init_type == "uniform":
            model.apply(weights_init_uniform_rule)
        elif init_type == "normal":
            model.apply(weights_init_normal_rule)
        elif init_type == "xavier_uniform":
            model.apply(weights_init_xavier_uniform_rule)
        elif init_type == "xavier_normal":
            model.apply(weights_init_xavier_normal_rule)
        elif init_type == "kaiming_uniform":
            model.apply(weights_init_kaiming_uniform_rule)
        elif init_type == "kaiming_normal":
            model.apply(weights_init_kaiming_normal_rule)
            
        model = model.to(device)
        print("==> Architecture: {}\n{}".format(model.name, model))
        print("==> Trainable parameters: {}".format(count_parameters(model)))
        
        # set optimizer
        if optimizer_choice == "SGD":
            optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)
        elif optimizer_choice == "Adam":
            optimizer = optim.Adam(model.parameters())
        elif optimizer_choice == "Adadelta":
            optimizer = optim.Adadelta(model.parameters())
        elif optimizer_choice == "RMSprop":
            optimizer = optim.RMSprop(model.parameters())
        
        # set loss
        if loss_choice == "l2":
            criterion = nn.MSELoss(reduction="mean")
        elif loss_choice == "l1":
            criterion = nn.L1Loss(reduction="mean")
        elif loss_choice == "lfk":
            criterion = FKLoss()
        
        
        print("\n==> Experiment {} Training network: {}".format(experiment_number+1, model.name))
        print("==> Training device: {}".format(device))
        

        # create a directory to save weights
        save_path = "results/"+robot_choice+"/"+robot_choice+"_" \
                    +model.name.replace(" ","").replace("[","_").replace("]","_").replace(",","-") \
                    +optimizer_choice+"_"+loss_choice+"_"+str(experiment_number+1)+'_qlim_scale_'+str(int(scale))

        if not os.path.exists(save_path):
            os.makedirs(save_path)




        if save_option == "local":

            # save test sets        
            df = pd.DataFrame(X_test)
            df.to_csv(save_path+"/X_test_"+robot_choice+"_"+str(dataset_samples)+".csv",
                index=False,
                header=pose_header)   

            df = pd.DataFrame(y_test)
            df.to_csv(save_path+"/y_test_"+robot_choice+"_"+str(dataset_samples)+".csv",
                index=False,
                header=joint_header)
        
        elif save_option == "cloud":    
            run = wandb.init(
                project = "iksolver-experiments",                # set the project name this run will be logged
                name = "Model_"+robot_choice+"_" \
                        +model.name.replace(" ","").replace("[","_").replace("]","_").replace(",","-") \
                        +optimizer_choice+"_"+loss_choice+"_"+str(experiment_number+1)+'_qlim_scale_'+str(int(scale))
            )
        


        
        train_losses = []
        valid_losses = []
        all_losses = []
        best_valid_loss = float('inf')
        start_time_train = time.monotonic()
        start_time = time.monotonic()
        for epoch in range(EPOCHS):
            
            train_loss = train(model, train_data_loader, optimizer, criterion, loss_choice, batch_size, device, epoch, EPOCHS)        
            valid_loss = evaluate(model, test_data_loader, criterion, loss_choice, device, epoch, EPOCHS)
        
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            all_losses.append([train_loss, valid_loss])

            train_metrics= {
                "train/epoch": epoch,
                "train/train_loss": train_loss,
            }
            
        
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_epoch = epoch

                if save_option == "local":
                    torch.save(model.state_dict(), save_path+'/best_epoch.pth')
                elif save_option == "cloud":
                    #torch.save(model.state_dict(), save_path+'/best_epoch.pth')
                    torch.save(model.state_dict(), save_path+'/best_epoch.pth')
                    artifact = wandb.Artifact(name="Model_"+robot_choice+"_" \
                                                    +model.name.replace(" ","").replace("[","_").replace("]","_").replace(",","-") \
                                                    +optimizer_choice+"_"+loss_choice+"_"+str(experiment_number+1)+'_qlim_scale_'+str(int(scale)), 
                                              type='model')
                    artifact.add_file(save_path+'/best_epoch.pth')
                    run.log_artifact(artifact)

            
            
            val_metrics = {
                "val/val_loss": valid_loss,
            }
            wandb.log({**train_metrics, **val_metrics})



            
            if epoch % (EPOCHS/print_steps) == 0 or epoch == EPOCHS-1:
            #if epoch % (1) == 0 or epoch == EPOCHS-1:
                if print_epoch:
                    end_time = time.monotonic()
                    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
                    print('Epoch: {}/{} | Epoch Time: {}m {}s'.format(epoch, EPOCHS, epoch_mins, epoch_secs))
                    print('\tTrain Loss: {}'.format(train_loss))
                    print('\tValid Loss: {}'.format(valid_loss))
                    print("\tBest Epoch Occurred [{}/{}]".format(best_epoch, EPOCHS)) 
                
                if save_option == "local":   
                    torch.save(model.state_dict(), save_path+'/epoch_'+str(epoch)+'.pth')
                elif save_option == "cloud":
                    torch.save(model.state_dict(), os.path.join(wandb.run.dir, "epoch_"+str(epoch)+".pth"))
    
                # save the histories of losses
                header = ["train loss", "valid loss"]
                
                df = pd.DataFrame(np.array(all_losses))
                df.to_csv(save_path+"/losses_"+robot_choice+"_"+str(dataset_samples)+".csv",
                    index=False,
                    header=header)
                
                end_time_train = time.monotonic()
                epoch_mins, epoch_secs = epoch_time(start_time_train, end_time_train)
        
        if print_epoch:
            print('\nEnd of Training for {} - Elapsed Time: {}m {}s'.format(model.name, epoch_mins, epoch_secs))    

        
    
    #print("Resetting the architecture ...\n\n")
    #hidden_layer_sizes = np.zeros((1,layers))
    wandb.finish()


            
                



