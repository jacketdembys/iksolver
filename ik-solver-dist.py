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
from models import *




def fit_model(model,
              input_dim,
              train_data_loader,
              test_data_loader,
              X_test,
              y_test,
              learning_rate,
              optimizer_choice,
              optimizer, 
              criterion,
              loss_choice, 
              batch_size, 
              device,
              EPOCHS,
              robot_choice,
              network_type,
              neurons,
              knowledge_phase,
              save_layers_str):
    
    print("\n\n==> Training for Knowledge phase: {}".format(knowledge_phase))

    # create a directory to save weights
    save_path = "results/"+robot_choice+"/"+network_type+"_"+knowledge_phase+"_"+robot_choice+"_" \
                + save_layers_str + "_neurons_" + str(neurons) + "_batch_" + str(batch_size)  +"_" \
                +optimizer_choice+"_"+loss_choice+"_"+str(experiment_number)+'_qlim_scale_'+str(int(scale))+'_samples_'+str(dataset_samples)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

 

    # save test sets 
    """
    if save_option == "cloud":
        run = wandb.init(
            project = "ik-0",                
            group = network_type+"_Dist_"+"Dataset_"+str(dataset_samples)+"_Scale_"+str(int(scale))+"_"+dataset_type+"_"+loss_choice,  # "_seq", "_1_to_1"
            #group = "Dataset_Scale_"+str(int(scale)),
            name = "model_"+robot_choice+"_" \
                    + save_layers_str + "_neurons_" + str(neurons) + "_batch_" + str(batch_size) +"_" \
                    +optimizer_choice+"_"+loss_choice+"_run_"+str(experiment_number)+'_qlim_scale_'+str(int(scale))+'_samples_'+str(dataset_samples)  #+'_non_traj_split', '_traj_split'   
            #name = "Model_"+robot_choice+"_" \
            #        +model.name.replace(" ","").replace("[","_").replace("]","_").replace(",","-") \
            #        +optimizer_choice+"_"+loss_choice+"_run_"+str(experiment_number)+'_qlim_scale_'+str(int(scale))+'_samples_'+str(dataset_samples)
        )
    """

    scaler = torch.cuda.amp.GradScaler()
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, steps_per_epoch=len(train_data_loader_priv), epochs=EPOCHS)  
    #patience = 0.1*EPOCHS
    patience = 100
    train_losses = []
    valid_losses = []
    all_losses = []
    best_valid_loss = float('inf')
    start_time_train = time.monotonic()
    start_time = time.monotonic()
    for epoch in range(EPOCHS):        
        
        train_loss = train(model, train_data_loader, optimizer, criterion, loss_choice, batch_size, device, epoch, EPOCHS, scheduler, scaler)        
        valid_loss = evaluate(model, test_data_loader, criterion, loss_choice, device, epoch, EPOCHS)
    
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        all_losses.append([train_loss, valid_loss])


        #if save_option == "cloud":
        train_metrics= {
            "train/epoch": epoch,
            "train/train_loss": train_loss,
        }
    
        val_metrics = {
            "val/val_loss": valid_loss,
        }
        wandb.log({**train_metrics, **val_metrics})
        #wandb.watch(model, criterion, log="all")
        
    
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_epoch = epoch
            counter = 0



            #torch.save(model.state_dict(), save_path+'/best_epoch.pth')
            if save_option == "local":
                torch.save(model.state_dict(), save_path+'/best_epoch_'+knowledge_phase+'.pth')
            
            elif save_option == "cloud":
                torch.save(model.state_dict(), save_path+'/best_epoch_'+knowledge_phase+'.pth')
                """
                artifact = wandb.Artifact(name=knowledge_phase+"_model_"+robot_choice+"_" \
                                                +model.name.replace(" ","").replace("[","_").replace("]","_").replace(",","-") \
                                                +optimizer_choice+"_"+loss_choice+"_"+str(experiment_number+1)+'_qlim_scale_'+str(int(scale)), 
                                            type='model')
                artifact.add_file(save_path+'/best_epoch_'+knowledge_phase+'.pth')
                run.log_artifact(artifact)
                """
                

        else:
            counter += 1
            if counter >= patience:
                print("Early stopping at epoch {}, best epoch: {}".format(epoch, best_epoch))
                break



        
        if epoch % (EPOCHS/print_steps) == 0 or epoch == EPOCHS-1:
        #if epoch % (1) == 0 or epoch == EPOCHS-1:
            if print_epoch:
                end_time = time.monotonic()
                epoch_mins, epoch_secs = epoch_time(start_time, end_time)
                print('\nEpoch: {}/{} | Epoch Time: {}m {}s'.format(epoch, EPOCHS, epoch_mins, epoch_secs))
                print('\tTrain Loss: {}'.format(train_loss))
                print('\tValid Loss: {}'.format(valid_loss))
                print("\tBest Epoch Occurred [{}/{}]".format(best_epoch, EPOCHS)) 
            
            if save_option == "local":   
                torch.save(model.state_dict(), save_path+'/epoch_'+str(epoch)+'.pth')
            elif save_option == "cloud":
                torch.save(model.state_dict(), save_path+'/epoch_'+str(epoch)+'.pth')
                #artifact2 = wandb.Artifact(name="Model_"+robot_choice+"_" \
                #                                +model.name.replace(" ","").replace("[","_").replace("]","_").replace(",","-") \
                #                                +optimizer_choice+"_"+loss_choice+"_"+str(experiment_number+1)+'_qlim_scale_'+str(int(scale)), 
                #                            type='model')
                #artifact2.add_file(save_path+'/epoch_'+str(epoch)+'.pth')
                #run.log_artifact(artifact2)
                #torch.save(model.state_dict(), os.path.join(wandb.run.dir, "epoch_"+str(epoch)+".pth"))

            # save the histories of losses
            header = ["train loss", "valid loss"]
            
            df = pd.DataFrame(np.array(all_losses))
            df.to_csv(save_path+"/"+knowledge_phase+"_losses_"+robot_choice+"_"+str(dataset_samples)+".csv",
                index=False,
                header=header)
                      
            
    end_time_train = time.monotonic()
    epoch_mins, epoch_secs = epoch_time(start_time_train, end_time_train)
    
    if print_epoch:
        print('\nEnd of Training for {} with {} phase - Elapsed Time: {}m {}s'.format(model.name, knowledge_phase, epoch_mins, epoch_secs))    

    


    ##############################################################################################################
    # Inference
    ##############################################################################################################
    # training is done, let's run inferences and record the evaluation metrics
    print("\n\n==> Testing the trained model ...")
    test_data_loader = load_test_dataset(X_test, y_test, device)
    weights_file = save_path+"/best_epoch_"+knowledge_phase+".pth"

    if network_type == "MLP":
        model = MLP(input_dim, hidden_layer_sizes, output_dim).to(device)
        #model = MLP(mapping_size*2, hidden_layer_sizes, output_dim).to(device)
    elif network_type == "ResMLP":
        model = ResMLPSum(input_dim, neurons, output_dim, num_blocks).to(device)
    elif network_type == "DenseMLP":
        model = DenseMLP(input_dim, neurons, output_dim, num_blocks).to(device)
    
    state_dict = model.state_dict()
    for n, p in torch.load(weights_file, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)
    
    
    # get the results from testing    
    with torch.no_grad():
        results = inference(model, test_data_loader, criterion, device, robot_choice)

    X_errors = results["X_errors"]
    
    # get some inference stats
    X_errors_r = X_errors[:,:6]
    X_errors_r[:,:3] = X_errors_r[:,:3] * 1000
    X_errors_r[:,3:] = np.rad2deg(X_errors_r[:,3:]) 
    avg_position_error = X_errors_r[1,:3].mean()
    avg_orientation_error = X_errors_r[1,3:].mean()

    print("avg_position_error (mm): {}".format(avg_position_error))
    print("avg_orientation_error (deg): {}".format(avg_orientation_error))


    X_preds = results["X_preds"]
    X_desireds = results["X_desireds"]
    X_errors_p = np.abs(X_preds - X_desireds)
    X_errors_p[:,:3] = X_errors_p[:,:3] * 1000
    X_errors_p[:,3:] = np.rad2deg(X_errors_p[:,3:]) 
    X_percentile = stats.percentileofscore(X_errors_p[:,0], [1,5,10,15,20], kind='rank')
    Y_percentile = stats.percentileofscore(X_errors_p[:,1], [1,5,10,15,20], kind='rank')
    Z_percentile = stats.percentileofscore(X_errors_p[:,2], [1,5,10,15,20], kind='rank')
    Ro_percentile = stats.percentileofscore(X_errors_p[:,3], [1,2,3,4,5], kind='rank')
    Pi_percentile = stats.percentileofscore(X_errors_p[:,4], [1,2,3,4,5], kind='rank')
    Ya_percentile = stats.percentileofscore(X_errors_p[:,5], [1,2,3,4,5], kind='rank')

    #print(X_errors_p.shape)
    #print(X_errors_r.shape)
    #print(X_errors_r)
    #print(model.name)




    # log this dataframe to wandb
    #columns = ["trainLoss", "validLoss"]
    #df2 = pd.DataFrame(np.array(all_losses))
    inference_results = {
        "device_name": device_name,
        "data_size": dataset_samples,
        "knowledge_phase": knowledge_phase,
        "joints_scale": scale,
        "architecture": model.name,
        "network": network_type,
        "layers": layers,
        "neurons": neurons,
        "optimizer": optimizer_choice,
        "loss": loss_choice,
        "completed_epochs": epoch,
        "best_epoch": best_epoch,
        "elapsed_time": "{}m {}s".format(epoch_mins, epoch_secs),
        "average_position_error(mm)": avg_position_error,
        "average_orientation_error(deg)": avg_orientation_error,
        "min_x(mm)": X_errors_r[0,0],
        "avg_x(mm)": X_errors_r[1,0],
        "max_x(mm)": X_errors_r[2,0],
        "std_x(mm)": X_errors_r[3,0],
        "x_percent_1(mm)": X_percentile[0],
        "x_percent_5(mm)": X_percentile[1],
        "x_percent_10(mm)": X_percentile[2],
        "x_percent_15(mm)": X_percentile[3],
        "x_percent_20(mm)": X_percentile[4],
        "min_y(mm)": X_errors_r[0,1],
        "avg_y(mm)": X_errors_r[1,1],
        "max_y(mm)": X_errors_r[2,1],
        "std_y(mm)": X_errors_r[3,1],
        "y_percent_1(mm)": Y_percentile[0],
        "y_percent_5(mm)": Y_percentile[1],
        "y_percent_10(mm)": Y_percentile[2],
        "y_percent_15(mm)": Y_percentile[3],
        "y_percent_20(mm)": Y_percentile[4],
        "min_z(mm)": X_errors_r[0,2],
        "avg_z(mm)": X_errors_r[1,2],
        "max_z(mm)": X_errors_r[2,2],
        "std_z(mm)": X_errors_r[3,2],
        "Z_percent_1(mm)": Z_percentile[0],
        "Z_percent_5(mm)": Z_percentile[1],
        "Z_percent_10(mm)": Z_percentile[2],
        "Z_percent_15(mm)": Z_percentile[3],
        "Z_percent_20(mm)": Z_percentile[4],
        "min_ro(deg)": X_errors_r[0,3],
        "avg_ro(deg)": X_errors_r[1,3],
        "max_ro(deg)": X_errors_r[2,3],
        "std_ro(deg)": X_errors_r[3,3],
        "ro_percent_1(deg)": Ro_percentile[0],
        "ro_percent_2(deg)": Ro_percentile[1],
        "ro_percent_3(deg)": Ro_percentile[2],
        "ro_percent_4(deg)": Ro_percentile[3],
        "ro_percent_5(deg)": Ro_percentile[4],
        "min_pi(deg)": X_errors_r[0,4],
        "avg_pi(deg)": X_errors_r[1,4],
        "max_pi(deg)": X_errors_r[2,4],
        "std_pi(deg)": X_errors_r[3,4],
        "pi_percent_1(deg)": Pi_percentile[0],
        "pi_percent_2(deg)": Pi_percentile[1],
        "pi_percent_3(deg)": Pi_percentile[2],
        "pi_percent_4(deg)": Pi_percentile[3],
        "pi_percent_5(deg)": Pi_percentile[4],
        "min_ya(deg)": X_errors_r[0,5],
        "avg_ya(deg)": X_errors_r[1,5],
        "max_ya(deg)": X_errors_r[2,5],
        "std_ya(deg)": X_errors_r[3,5],
        "ya_percent_1(deg)": Ya_percentile[0],
        "ya_percent_2(deg)": Ya_percentile[1],
        "ya_percent_3(deg)": Ya_percentile[2],
        "ya_percent_4(deg)": Ya_percentile[3],
        "ya_percent_5(deg)": Ya_percentile[4],
    }
    inference_results = pd.DataFrame(inference_results, index=[0])
    inference_results_table = wandb.Table(data=inference_results)
    wandb.log({"inferences_"+knowledge_phase: inference_results_table})


    return model






if __name__ == '__main__':

    
    print('==> Reading from the config file ...')

    # Read parameters from configuration file
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path",
                        type=str,
                        default="train.yaml",
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
    num_blocks =  config["MODEL"]["NUM_BLOCKS"]     
    dataset_samples = config["TRAIN"]["DATASET"]["NUM_SAMPLES"]                   # MLP, ResMLP, DenseMLP, FouierMLP 
    print_steps = config["TRAIN"]["PRINT_STEPS"] 
    save_option = config["TRAIN"]["CHECKPOINT"]["SAVE_OPTIONS"]                                # local or cloud
    load_option = config["TRAIN"]["CHECKPOINT"]["LOAD_OPTIONS"]  
    dataset_type = config["TRAIN"]["DATASET"]["TYPE"]

    scale = config["TRAIN"]["DATASET"]["JOINT_LIMIT_SCALE"]
    EPOCHS = config["TRAIN"]["HYPERPARAMETERS"]["EPOCHS"]                         # total training epochs   

    
    if save_option == "cloud":
        print('==> Log in to wandb to send out metrics ...')
        wandb.login()                                        # login to the Weights and Biases   
               
    
    print("==> Running based on configuration...")
    
    #device = torch.device('cpu')
    #device_name = "cpu"
    #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')         # device to train on
    device = torch.device('cuda:'+str(config["DEVICE_ID"]) if torch.cuda.is_available() else 'cpu') 
    #device_name = "cpu"
    device_name = torch.cuda.get_device_name(device)
    
    # set input and output size based on robot
    if robot_choice == "6DoF-6R-Puma260":
        n_DoF = 6
        input_dim = 6
        output_dim = 6
        pose_header = ["x", "y", "z","R","P","Y"]
        joint_header = ["t1", "t2", "t3", "t4", "t5", "t6"]
    if robot_choice == "7DoF-7R-Panda":
        if dataset_type == "1_to_1":
            n_DoF = 7
            input_dim = 6 #+6+7 #6
            output_dim = 7
        elif dataset_type == "seq":
            n_DoF = 7
            input_dim_priv = 6+6+7 #6
            input_dim_reg = 6
            output_dim = 7

        pose_header = ["x", "y", "z","R","P","Y"]
        joint_header = ["t1", "t2", "t3", "t4", "t5", "t6", "t7"]
        
    # load dataset from file
    if load_option == "cloud":
        if dataset_type == "1_to_1":
            data = pd.read_csv('/home/datasets/'+robot_choice+'/data_'+robot_choice+'_'+str(int(dataset_samples))+'_qlim_scale_'+str(int(scale))+'.csv')
        elif dataset_type == "seq":
            data = pd.read_csv('/home/datasets/'+robot_choice+'/data_'+robot_choice+'_'+str(int(dataset_samples))+'_qlim_scale_'+str(int(scale))+'_seq.csv')
    elif load_option == "local":
        if dataset_type == "1_to_1":
            data = pd.read_csv('../docker/datasets/'+robot_choice+'/data_'+robot_choice+'_'+str(int(dataset_samples))+'_qlim_scale_'+str(int(scale))+'.csv')
        elif dataset_type == "seq":
            data = pd.read_csv('../docker/datasets/'+robot_choice+'/data_'+robot_choice+'_'+str(int(dataset_samples))+'_qlim_scale_'+str(int(scale))+'_seq.csv')
    data_a = np.array(data) 




    # number of times the experiments needs to be repeated 
    # read these variabkkes from outside the configuration file
    experiments = config["NUM_EXPERIMENT_REPETITIONS"]
    layers = config["MODEL"]["NUM_HIDDEN_LAYERS"]
    neurons = config["MODEL"]["NUM_HIDDEN_NEURONS"]   
    #layers = args.neurons
    #neurons = args.layers   
    #scale = args.scale


    # set the hidden layer array to initialize the architecture
    hidden_layer_sizes = np.zeros((1,layers))          
    
    #for neuron in range(200, neurons+100, 100):


    hidden_layer_sizes[:,:] = neurons
    hidden_layer_sizes = hidden_layer_sizes.squeeze(0).astype(int).tolist()

    
    #for experiment_number in range(experiments):
    experiment_number = experiments

    # ensure reproducibilities if seed is set to true
    if seed_choice:   
        random.seed(seed_number)
        np.random.seed(seed_number)
        torch.manual_seed(seed_number)
        torch.cuda.manual_seed(seed_number)
        torch.backends.cudnn.deterministic = True
    ## train and validate
    # load the dataset with privilege information
    dataset_type = "seq"
    train_data_loader_priv, test_data_loader_priv, X_validate_priv, y_validate_priv, X_train_priv, y_train_priv, X_test_priv, y_test_priv = load_dataset(data_a, n_DoF, batch_size, robot_choice, dataset_type, device)

    # load the dataset without privilege information
    dataset_type = "1_to_1"
    data_reg = data_a[:,13:]
    train_data_loader_reg, test_data_loader_reg, X_validate_reg, y_validate_reg, X_train_reg, y_train_reg, X_test_reg, y_test_reg = load_dataset(data_reg, n_DoF, batch_size, robot_choice, dataset_type, device)

    
    """
    for data1, data2 in zip(train_data_loader_priv, train_data_loader_reg): 
        x1, y1 = data1['input'], data1['output']
        x2, y2 = data2['input'], data2['output']
        print(x1[0,:],y1[0,:])
        print(x2[0,:],y2[0,:])
        sys.exit()
    """

    #print(input_dim)
    #print(hidden_layer_sizes)
    #print(output_dim)

    
    # get network architecture
    model_priv = DenseMLP(input_dim_priv, neurons, output_dim, num_blocks)
    model_reg = DenseMLP(input_dim_reg, neurons, output_dim, num_blocks)
    model_dist = DenseMLP(input_dim_reg, neurons, output_dim, num_blocks)
    save_layers_str = "blocks_"+ str(num_blocks)

    """
    if network_type == "MLP":
        model = MLP(input_dim, hidden_layer_sizes, output_dim)
        save_layers_str = "layers_"+ str(layers)
        #model = MLP(mapping_size*2, hidden_layer_sizes, output_dim)
    elif network_type == "ResMLP":
        model = ResMLPSum(input_dim, neurons, output_dim, num_blocks)
        save_layers_str = "blocks_"+ str(num_blocks)
    elif network_type == "DenseMLP":
        model_priv = DenseMLP(input_dim, neurons, output_dim, num_blocks)
        model_reg = DenseMLP(input_dim, neurons, output_dim, num_blocks)
        model_dist = DenseMLP(input_dim, neurons, output_dim, num_blocks)
        save_layers_str = "blocks_"+ str(num_blocks)
    elif network_type == "FourierMLP":
        fourier_dim = 16
        scale = 10
        model = FourierMLP(input_dim, fourier_dim, hidden_layer_sizes, output_dim, scale)
    """
        
    
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
        
    model_priv = model_priv.to(device)
    model_reg = model_reg.to(device)
    model_dist = model_dist.to(device)
    print("==> Teacher Architecture: {}\n{}".format(model_priv.name, model_priv))
    print("==> Trainable parameters: {}".format(count_parameters(model_priv)))

    print("==> Student Architecture: {}\n{}".format(model_dist.name, model_dist))
    print("==> Trainable parameters: {}".format(count_parameters(model_dist)))
    
    # set optimizer
    if optimizer_choice == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)
    elif optimizer_choice == "Adam":
        optimizer_priv = optim.Adam(model_priv.parameters())
        optimizer_reg = optim.Adam(model_reg.parameters())
        optimizer_dist = optim.Adam(model_dist.parameters())
    elif optimizer_choice == "Adadelta":
        optimizer = optim.Adadelta(model.parameters())
    elif optimizer_choice == "RMSprop":
        optimizer = optim.RMSprop(model.parameters())
    
    # set loss
    if loss_choice == "lq":
        criterion = nn.MSELoss(reduction="mean")
    elif loss_choice == "l1":
        criterion = nn.L1Loss(reduction="mean")
    elif loss_choice == "ld":
        criterion = FKLoss(robot_choice=robot_choice, device=device)
    
    
    # save test sets 
    if save_option == "cloud":
        run = wandb.init(
            project = "ik-0",                
            group = network_type+"_Dist_"+"Dataset_"+str(dataset_samples)+"_Scale_"+str(int(scale))+"_"+dataset_type+"_"+loss_choice,  # "_seq", "_1_to_1"
            #group = "Dataset_Scale_"+str(int(scale)),
            name = "model_"+robot_choice+"_" \
                    + save_layers_str + "_neurons_" + str(neurons) + "_batch_" + str(batch_size) +"_" \
                    +optimizer_choice+"_"+loss_choice+"_run_"+str(experiment_number)+'_qlim_scale_'+str(int(scale))+'_samples_'+str(dataset_samples)  #+'_non_traj_split', '_traj_split'   
            #name = "Model_"+robot_choice+"_" \
            #        +model.name.replace(" ","").replace("[","_").replace("]","_").replace(",","-") \
            #        +optimizer_choice+"_"+loss_choice+"_run_"+str(experiment_number)+'_qlim_scale_'+str(int(scale))+'_samples_'+str(dataset_samples)
        )


    print("\n==> Experiment {} Training network: {}".format(experiment_number, model_priv.name))
    print("==> Training device: {}".format(device))
    

    ##############################################################################################################
    # Training and Validation
    ############################################################################################################## 
    
    # Train network with privilege information
    knowledge_phase = "priv"
    model_priv = fit_model(model_priv,
              input_dim_priv,
              train_data_loader_priv,
              test_data_loader_priv,
              X_test_priv,
              y_test_priv,
              learning_rate,
              optimizer_choice,
              optimizer_priv, 
              criterion,
              loss_choice, 
              batch_size, 
              device,
              EPOCHS,
              robot_choice,
              network_type,
              neurons,
              knowledge_phase,
              save_layers_str)

    """
    knowledge_phase = "reg"
    model_reg = fit_model(model_reg,
              input_dim_reg,
              train_data_loader_reg,
              test_data_loader_reg,
              X_test_reg,
              y_test_reg,
              learning_rate,
              optimizer_choice,
              optimizer_reg, 
              criterion,
              loss_choice, 
              batch_size, 
              device,
              EPOCHS,
              robot_choice,
              network_type,
              neurons,
              knowledge_phase,
              save_layers_str)

    """

    knowledge_phase = "dist"
    print("\n\n==> Training for Knowledge phase: {}".format(knowledge_phase))

    # create a directory to save weights
    save_path = "results/"+robot_choice+"/"+network_type+"_"+knowledge_phase+"_"+robot_choice+"_" \
                + save_layers_str + "_neurons_" + str(neurons) + "_batch_" + str(batch_size)  +"_" \
                +optimizer_choice+"_"+loss_choice+"_"+str(experiment_number)+'_qlim_scale_'+str(int(scale))+'_samples_'+str(dataset_samples)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

 
    # freeze the layers in model_priv
    for param in model_priv.parameters():
        param.requires_grad =False

    scaler = torch.cuda.amp.GradScaler()
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer_dist, max_lr=learning_rate, steps_per_epoch=len(train_data_loader_priv), epochs=EPOCHS)  
    #patience = 0.1*EPOCHS
    patience = 100
    alpha = 1
    train_losses = []
    valid_losses = []
    all_losses = []
    best_valid_loss = float('inf')
    start_time_train = time.monotonic()
    start_time = time.monotonic()
    for epoch in range(EPOCHS):        
        
        train_loss = train_dist(model_priv, model_dist, train_data_loader_priv, train_data_loader_reg, optimizer_dist, criterion, loss_choice, batch_size, device, epoch, EPOCHS, scheduler, scaler, alpha)        
        valid_loss = evaluate(model_dist, test_data_loader_reg, criterion, loss_choice, device, epoch, EPOCHS)
    
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        all_losses.append([train_loss, valid_loss])


        #if save_option == "cloud":
        train_metrics= {
            "train/epoch": epoch,
            "train/train_loss": train_loss,
        }
    
        val_metrics = {
            "val/val_loss": valid_loss,
        }
        wandb.log({**train_metrics, **val_metrics})
        #wandb.watch(model, criterion, log="all")
        
    
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_epoch = epoch
            counter = 0



            #torch.save(model.state_dict(), save_path+'/best_epoch.pth')
            if save_option == "local":
                torch.save(model_dist.state_dict(), save_path+'/best_epoch'+knowledge_phase+'.pth')
            
            elif save_option == "cloud":
                torch.save(model_dist.state_dict(), save_path+'/best_epoch'+knowledge_phase+'.pth')
                """
                artifact = wandb.Artifact(name=knowledge_phase+"_model_"+robot_choice+"_" \
                                                +model_dist.name.replace(" ","").replace("[","_").replace("]","_").replace(",","-") \
                                                +optimizer_choice+"_"+loss_choice+"_"+str(experiment_number+1)+'_qlim_scale_'+str(int(scale)), 
                                            type='model')
                artifact.add_file(save_path+'/best_epoch_'+knowledge_phase+'.pth')
                run.log_artifact(artifact)
                """
                

        else:
            counter += 1
            if counter >= patience:
                print("Early stopping at epoch {}, best epoch: {}".format(epoch, best_epoch))
                break



        
        if epoch % (EPOCHS/print_steps) == 0 or epoch == EPOCHS-1:
        #if epoch % (1) == 0 or epoch == EPOCHS-1:
            if print_epoch:
                end_time = time.monotonic()
                epoch_mins, epoch_secs = epoch_time(start_time, end_time)
                print('\nEpoch: {}/{} | Epoch Time: {}m {}s'.format(epoch, EPOCHS, epoch_mins, epoch_secs))
                print('\tTrain Loss: {}'.format(train_loss))
                print('\tValid Loss: {}'.format(valid_loss))
                print("\tBest Epoch Occurred [{}/{}]".format(best_epoch, EPOCHS)) 
            
            if save_option == "local":   
                torch.save(model_dist.state_dict(), save_path+'/epoch_'+str(epoch)+'.pth')
            elif save_option == "cloud":
                torch.save(model_dist.state_dict(), save_path+'/epoch_'+str(epoch)+'.pth')
                #artifact2 = wandb.Artifact(name="Model_"+robot_choice+"_" \
                #                                +model.name.replace(" ","").replace("[","_").replace("]","_").replace(",","-") \
                #                                +optimizer_choice+"_"+loss_choice+"_"+str(experiment_number+1)+'_qlim_scale_'+str(int(scale)), 
                #                            type='model')
                #artifact2.add_file(save_path+'/epoch_'+str(epoch)+'.pth')
                #run.log_artifact(artifact2)
                #torch.save(model.state_dict(), os.path.join(wandb.run.dir, "epoch_"+str(epoch)+".pth"))

            # save the histories of losses
            header = ["train loss", "valid loss"]
            
            df = pd.DataFrame(np.array(all_losses))
            df.to_csv(save_path+"/"+knowledge_phase+"_losses_"+robot_choice+"_"+str(dataset_samples)+".csv",
                index=False,
                header=header)
                      
            
    end_time_train = time.monotonic()
    epoch_mins, epoch_secs = epoch_time(start_time_train, end_time_train)
    
    if print_epoch:
        print('\nEnd of Training for {} with {} phase - Elapsed Time: {}m {}s'.format(model_dist.name, knowledge_phase, epoch_mins, epoch_secs))    

    


    ##############################################################################################################
    # Inference
    ##############################################################################################################
    # training is done, let's run inferences and record the evaluation metrics
    print("\n\n==> Testing the trained model ...")
    test_data_loader = load_test_dataset(X_test_reg, y_test_reg, device)
    weights_file = save_path+"/best_epoch"+knowledge_phase+".pth"

    if network_type == "MLP":
        model_dist = MLP(input_dim_reg, hidden_layer_sizes, output_dim).to(device)
        #model = MLP(mapping_size*2, hidden_layer_sizes, output_dim).to(device)
    elif network_type == "ResMLP":
        model_dist = ResMLPSum(input_dim_reg, neurons, output_dim, num_blocks).to(device)
    elif network_type == "DenseMLP":
        model_dist = DenseMLP(input_dim_reg, neurons, output_dim, num_blocks).to(device)
    
    state_dict = model_dist.state_dict()
    for n, p in torch.load(weights_file, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)
    
    
    # get the results from testing    
    with torch.no_grad():
        results = inference(model_dist, test_data_loader_reg, criterion, device, robot_choice)

    X_errors = results["X_errors"]
    
    # get some inference stats
    X_errors_r = X_errors[:,:6]
    X_errors_r[:,:3] = X_errors_r[:,:3] * 1000
    X_errors_r[:,3:] = np.rad2deg(X_errors_r[:,3:]) 
    avg_position_error = X_errors_r[1,:3].mean()
    avg_orientation_error = X_errors_r[1,3:].mean()

    print("avg_position_error (mm): {}".format(avg_position_error))
    print("avg_orientation_error (deg): {}".format(avg_orientation_error))




    X_preds = results["X_preds"]
    X_desireds = results["X_desireds"]
    X_errors_p = np.abs(X_preds - X_desireds)
    X_errors_p[:,:3] = X_errors_p[:,:3] * 1000
    X_errors_p[:,3:] = np.rad2deg(X_errors_p[:,3:]) 
    X_percentile = stats.percentileofscore(X_errors_p[:,0], [1,5,10,15,20], kind='rank')
    Y_percentile = stats.percentileofscore(X_errors_p[:,1], [1,5,10,15,20], kind='rank')
    Z_percentile = stats.percentileofscore(X_errors_p[:,2], [1,5,10,15,20], kind='rank')
    Ro_percentile = stats.percentileofscore(X_errors_p[:,3], [1,2,3,4,5], kind='rank')
    Pi_percentile = stats.percentileofscore(X_errors_p[:,4], [1,2,3,4,5], kind='rank')
    Ya_percentile = stats.percentileofscore(X_errors_p[:,5], [1,2,3,4,5], kind='rank')

    #print(X_errors_p.shape)
    #print(X_errors_r.shape)
    #print(X_errors_r)
    #print(model.name)




    # log this dataframe to wandb
    #columns = ["trainLoss", "validLoss"]
    #df2 = pd.DataFrame(np.array(all_losses))
    inference_results = {
        "device_name": device_name,
        "data_size": dataset_samples,
        "knowledge_phase": knowledge_phase,
        "joints_scale": scale,
        "architecture": model_dist.name,
        "network": network_type,
        "layers": layers,
        "neurons": neurons,
        "optimizer": optimizer_choice,
        "loss": loss_choice,
        "completed_epochs": epoch,
        "best_epoch": best_epoch,
        "elapsed_time": "{}m {}s".format(epoch_mins, epoch_secs),
        "average_position_error(mm)": avg_position_error,
        "average_orientation_error(deg)": avg_orientation_error,
        "min_x(mm)": X_errors_r[0,0],
        "avg_x(mm)": X_errors_r[1,0],
        "max_x(mm)": X_errors_r[2,0],
        "std_x(mm)": X_errors_r[3,0],
        "x_percent_1(mm)": X_percentile[0],
        "x_percent_5(mm)": X_percentile[1],
        "x_percent_10(mm)": X_percentile[2],
        "x_percent_15(mm)": X_percentile[3],
        "x_percent_20(mm)": X_percentile[4],
        "min_y(mm)": X_errors_r[0,1],
        "avg_y(mm)": X_errors_r[1,1],
        "max_y(mm)": X_errors_r[2,1],
        "std_y(mm)": X_errors_r[3,1],
        "y_percent_1(mm)": Y_percentile[0],
        "y_percent_5(mm)": Y_percentile[1],
        "y_percent_10(mm)": Y_percentile[2],
        "y_percent_15(mm)": Y_percentile[3],
        "y_percent_20(mm)": Y_percentile[4],
        "min_z(mm)": X_errors_r[0,2],
        "avg_z(mm)": X_errors_r[1,2],
        "max_z(mm)": X_errors_r[2,2],
        "std_z(mm)": X_errors_r[3,2],
        "Z_percent_1(mm)": Z_percentile[0],
        "Z_percent_5(mm)": Z_percentile[1],
        "Z_percent_10(mm)": Z_percentile[2],
        "Z_percent_15(mm)": Z_percentile[3],
        "Z_percent_20(mm)": Z_percentile[4],
        "min_ro(deg)": X_errors_r[0,3],
        "avg_ro(deg)": X_errors_r[1,3],
        "max_ro(deg)": X_errors_r[2,3],
        "std_ro(deg)": X_errors_r[3,3],
        "ro_percent_1(deg)": Ro_percentile[0],
        "ro_percent_2(deg)": Ro_percentile[1],
        "ro_percent_3(deg)": Ro_percentile[2],
        "ro_percent_4(deg)": Ro_percentile[3],
        "ro_percent_5(deg)": Ro_percentile[4],
        "min_pi(deg)": X_errors_r[0,4],
        "avg_pi(deg)": X_errors_r[1,4],
        "max_pi(deg)": X_errors_r[2,4],
        "std_pi(deg)": X_errors_r[3,4],
        "pi_percent_1(deg)": Pi_percentile[0],
        "pi_percent_2(deg)": Pi_percentile[1],
        "pi_percent_3(deg)": Pi_percentile[2],
        "pi_percent_4(deg)": Pi_percentile[3],
        "pi_percent_5(deg)": Pi_percentile[4],
        "min_ya(deg)": X_errors_r[0,5],
        "avg_ya(deg)": X_errors_r[1,5],
        "max_ya(deg)": X_errors_r[2,5],
        "std_ya(deg)": X_errors_r[3,5],
        "ya_percent_1(deg)": Ya_percentile[0],
        "ya_percent_2(deg)": Ya_percentile[1],
        "ya_percent_3(deg)": Ya_percentile[2],
        "ya_percent_4(deg)": Ya_percentile[3],
        "ya_percent_5(deg)": Ya_percentile[4],
    }
    inference_results = pd.DataFrame(inference_results, index=[0])
    inference_results_table = wandb.Table(data=inference_results)
    wandb.log({"inferences_"+knowledge_phase: inference_results_table})






    if save_option == "cloud":
        wandb.finish()


            
                



