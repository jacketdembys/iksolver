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
from models_2 import DenseNet


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
    orientation_type = config["TRAIN"]["DATASET"]["ORIENTATION"]

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
    if robot_choice == "7DoF-7R-Panda" or robot_choice == "7DoF-GP66":
        if dataset_type == "1_to_1":
            n_DoF = 7
            #input_dim = 6 #+6+7 #6
            #input_dim = 6 #+6+7 #6
            output_dim = 7
            if orientation_type == "RPY":  
                input_dim = 6 #6 
            elif orientation_type == "Quaternion": 
                input_dim = 7 
            elif orientation_type == "DualQuaternion": 
                input_dim = 8 
            elif orientation_type == "Rotation": 
                input_dim = 12 
            elif orientation_type == "Rotation6d": 
                input_dim = 9 
        elif dataset_type == "seq":
            n_DoF = 7
            input_dim = 6+6+7 #6
            output_dim = 7

        pose_header = ["x", "y", "z","R","P","Y"]
        joint_header = ["t1", "t2", "t3", "t4", "t5", "t6", "t7"]
    
    if robot_choice == "8DoF-P8":
        if dataset_type == "1_to_1":
            n_DoF = 8
            output_dim = 8
            if orientation_type == "RPY":  
                input_dim = 6 
            elif orientation_type == "Quaternion": 
                input_dim = 7 
            elif orientation_type == "DualQuaternion": 
                input_dim = 8 
            elif orientation_type == "Rotation": 
                input_dim = 12 
            elif orientation_type == "Rotation6d": 
                input_dim = 9 
        elif dataset_type == "seq":
            n_DoF = 8
            input_dim = 6+6+8 #6
            output_dim = 8

        pose_header = ["x", "y", "z","R","P","Y"]
        joint_header = ["t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8"]
        
    # load dataset from file
    if load_option == "cloud":
        if dataset_type == "1_to_1":
            data = pd.read_csv('/home/datasets/'+robot_choice+'/data_'+robot_choice+'_'+str(int(dataset_samples))+'_qlim_scale_'+str(int(scale))+'_seq.csv')
        elif dataset_type == "seq":
            data = pd.read_csv('/home/datasets/'+robot_choice+'/data_'+robot_choice+'_'+str(int(dataset_samples))+'_qlim_scale_'+str(int(scale))+'_seq.csv')
    elif load_option == "local":
        if dataset_type == "1_to_1":
            #data = pd.read_csv('../docker/datasets/'+robot_choice+'/data_'+robot_choice+'_'+str(int(dataset_samples))+'_qlim_scale_'+str(int(scale))+'_const.csv')
            data = pd.read_csv('../docker/datasets/'+robot_choice+'/data_'+robot_choice+'_'+str(int(dataset_samples))+'_qlim_scale_'+str(int(scale))+'_seq.csv') #+'_'+orientation_type+'.csv')
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
    # load the dataset
    train_data_loader, test_data_loader, X_validate, y_validate, X_train, y_train, X_test, y_test = load_dataset(data_a, n_DoF, batch_size, robot_choice, dataset_type, device, input_dim)

    #print(input_dim)
    #print(hidden_layer_sizes)
    #print(output_dim)

    
    # get network architecture
    if network_type == "MLP":

        ik_model = MLP(input_dim, hidden_layer_sizes, output_dim)
        fk_model = MLP(output_dim, hidden_layer_sizes, input_dim)
        save_layers_str = "blocks_"+ str(num_blocks)+"_layers_"+ str(layers)
        #model = MLP(mapping_size*2, hidden_layer_sizes, output_dim)

    elif network_type == "ResMLP":

        model = ResMLPSum(input_dim, neurons, output_dim, num_blocks)
        save_layers_str = "blocks_"+ str(num_blocks)+"_layers_"+ str(layers)

    elif network_type == "DenseMLP":

        model = DenseMLP(input_dim, neurons, output_dim, num_blocks)
        save_layers_str = "blocks_"+ str(num_blocks)+"_layers_"+ str(layers)

    elif network_type == "DenseMLP2":

        model = DenseMLP2(input_dim, neurons, output_dim, num_blocks)
        save_layers_str = "blocks_"+ str(num_blocks)+"_layers_"+ str(layers)

    elif network_type == "DenseMLP3":

        block_config = np.zeros((1,num_blocks))   
        block_config[:,:] = layers
        block_config = block_config.squeeze(0).astype(int).tolist()
        model = DenseNet(input_dim, neurons, block_config, output_dim)
        save_layers_str = "blocks_"+ str(num_blocks)+"_layers_"+ str(layers)


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
        
    ik_model = ik_model.to(device)
    fk_model = fk_model.to(device)
    print("==> Architecture: {}\n{}".format(ik_model.name, ik_model))
    print("==> Trainable parameters: {}".format(count_parameters(ik_model)))
    
    # set optimizer
    if optimizer_choice == "SGD":
        ik_optimizer = optim.SGD(ik_model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)
        #fk_optimizer = optim.SGD(fk_model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)

    elif optimizer_choice == "Adam":
        ik_optimizer = optim.Adam(list(ik_model.parameters())+list(fk_model.parameters()), lr=learning_rate)
        #ik_optimizer = optim.Adam(ik_model.parameters(), lr=learning_rate)
        #fk_optimizer = optim.Adam(fk_model.parameters(), lr=learning_rate)

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
        fk_criterion = FKLoss(robot_choice=robot_choice, device=device)
        ik_criterion = IKLoss(robot_choice=robot_choice, device=device)
    
    


    print("\n==> Experiment {} Training network: {}".format(experiment_number, ik_model.name))
    print("==> Training device: {}".format(device))
    

    # create a directory to save weights
    #save_path = "results/"+robot_choice+"/"+robot_choice+"_" \
    #            +model.name.replace(" ","").replace("[","_").replace("]","_").replace(",","-") \
    #            +optimizer_choice+"_"+loss_choice+"_"+str(experiment_number)+'_qlim_scale_'+str(int(scale))+'_samples_'+str(dataset_samples)
    
    # results_fkloss
    save_path = "results_two_stages/"+robot_choice+"/"+network_type+"_2_stages_"+robot_choice+"_" \
                + save_layers_str + "_neurons_" + str(neurons) + "_batch_" + str(batch_size)  +"_" \
                +optimizer_choice+"_"+loss_choice+"_"+str(experiment_number)+'_qlim_scale_'+str(int(scale))+'_samples_'+str(dataset_samples)+"_"+dataset_type+"_"+orientation_type+"_"+str(learning_rate)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

 

    # save test sets 
    """       
    df = pd.DataFrame(X_test)
    df.to_csv(save_path+"/X_test_"+robot_choice+"_"+str(dataset_samples)+".csv",
        index=False,
        header=pose_header)   

    df = pd.DataFrame(y_test)
    df.to_csv(save_path+"/y_test_"+robot_choice+"_"+str(dataset_samples)+".csv",
        index=False,
        header=joint_header)
    """

    # Modif_Err   Biternion    Modif_Err_2_  fkloss_Dataset_


    if save_option == "cloud":
        run = wandb.init(
            entity="jacketdembys",
            project = "ik-two-stages",                
            group = network_type+"_"+"Two_Stages_Dataset_"+str(dataset_samples)+"_Scale_"+str(int(scale))+"_"+dataset_type+"_"+loss_choice,  # "_seq", "_1_to_1"
            #group = "Dataset_Scale_"+str(int(scale)),
            name = "Two_Stages_Model_"+robot_choice+"_" \
                    + save_layers_str + "_neurons_" + str(neurons) + "_batch_" + str(batch_size) +"_" \
                    +optimizer_choice+"_"+loss_choice+"_run_"+str(experiment_number)+'_qlim_scale_'+str(int(scale))+'_samples_'+str(dataset_samples)+"_"+orientation_type+"_"+str(learning_rate)   #+'_non_traj_split', '_traj_split'   
            #name = "Model_"+robot_choice+"_" \
            #        +model.name.replace(" ","").replace("[","_").replace("]","_").replace(",","-") \
            #        +optimizer_choice+"_"+loss_choice+"_run_"+str(experiment_number)+'_qlim_scale_'+str(int(scale))+'_samples_'+str(dataset_samples)
        )


 

    ##############################################################################################################
    # Training and Validation
    ############################################################################################################## 
    scaler = torch.cuda.amp.GradScaler()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(ik_optimizer, mode='min', factor=0.1, patience=10, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=True)
    #scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, steps_per_epoch=len(train_data_loader), epochs=EPOCHS)  
    #patience = 0.1*EPOCHS
    patience = 100
    train_losses_fk = []
    valid_losses_fk = []
    train_losses_ik = []
    valid_losses_ik = []
    all_losses = []
    best_valid_loss = float('inf')
    best_valid_loss_fk = float('inf')
    best_valid_loss_ik = float('inf')
    start_time_train = time.monotonic()
    start_time = time.monotonic()

    for epoch in range(EPOCHS):        
        
        combined_train_loss, train_loss_fk, train_loss_ik = train_two_stages(fk_model, 
                                                                            ik_model, 
                                                                            train_data_loader, 
                                                                            #fk_optimizer,  
                                                                            ik_optimizer, 
                                                                            fk_criterion, 
                                                                            ik_criterion, 
                                                                            batch_size, 
                                                                            device, 
                                                                            epoch, 
                                                                            EPOCHS, 
                                                                            scheduler, 
                                                                            scaler,
                                                                            robot_choice)

        combined_valid_loss, valid_loss_fk, valid_loss_ik = evaluate_two_stages(fk_model, 
                                                                                ik_model,  
                                                                                test_data_loader, 
                                                                                fk_criterion, 
                                                                                ik_criterion, 
                                                                                device, 
                                                                                epoch, 
                                                                                EPOCHS, 
                                                                                robot_choice)
    
        #print(train_loss)
        #print(valid_loss)
        #sys.exit()

        #scheduler.step(valid_loss)

        train_losses_fk.append(train_loss_fk)
        valid_losses_fk.append(valid_loss_fk)
        train_losses_ik.append(train_loss_ik)
        valid_losses_ik.append(valid_loss_ik)
        all_losses.append([train_loss_fk, valid_loss_fk, train_loss_ik, valid_loss_ik])


        if save_option == "cloud":
            train_metrics= {
                "train/epoch": epoch,
                "train/train_loss": combined_train_loss,
                "train/train_loss_fk": train_loss_fk,
                "train/train_loss_ik": train_loss_ik
            }
        
            val_metrics = {
                "val/val_loss_fk": valid_loss_fk,
                "val/val_loss_ik": valid_loss_ik,
                "val/valid_loss": combined_valid_loss,
                "val/best_valid_loss_fk": best_valid_loss_fk,
                "val/best_valid_loss_ik": best_valid_loss_ik
            }
            wandb.log({**train_metrics, **val_metrics})
            #wandb.watch(model, criterion, log="all")
        
        #print(valid_loss)
        if valid_loss_fk < best_valid_loss_fk:
            best_valid_loss_fk = valid_loss_fk

            #torch.save(model.state_dict(), save_path+'/best_epoch.pth')
            if save_option == "local":
                
                torch.save(fk_model.state_dict(), save_path+'/best_epoch_fk.pth')
                """
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': fk_model.state_dict(),
                    'optimizer_state_dict': fk_optimizer.state_dict(),
                    'loss': train_loss_fk,
                    }, save_path+'/best_epoch_fk.pt')
                """  
            elif save_option == "cloud":
            
                torch.save(fk_model.state_dict(), save_path+'/best_epoch_fk.pth')
                """
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': fk_model.state_dict(),
                    'optimizer_state_dict': fk_optimizer.state_dict(),
                    'loss': train_loss_fk,
                    }, save_path+'/best_epoch_fk.pt')
                """


        if valid_loss_ik < best_valid_loss_ik:
            best_valid_loss_ik = valid_loss_ik       
            best_epoch = epoch
            counter = 0

            #torch.save(model.state_dict(), save_path+'/best_epoch.pth')
            if save_option == "local":
                
                torch.save(ik_model.state_dict(), save_path+'/best_epoch_ik.pth')
                """
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': ik_model.state_dict(),
                    'optimizer_state_dict': ik_optimizer.state_dict(),
                    'loss': train_loss_ik,
                    }, save_path+'/best_epoch_ik.pt')
                """    
            elif save_option == "cloud":

                torch.save(ik_model.state_dict(), save_path+'/best_epoch_ik.pth')
                """
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': ik_model.state_dict(),
                    'optimizer_state_dict': ik_optimizer.state_dict(),
                    'loss': train_loss_ik,
                    }, save_path+'/best_epoch_ik.pt')
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
                print('\tTrain Loss: {}'.format(combined_train_loss))
                print('\tValid Loss: {}\n'.format(combined_valid_loss))
                print('\tTrain Loss IK: {}'.format(train_loss_ik))
                print('\tValid Loss IK: {}\n'.format(valid_loss_ik))
                print('\tTrain Loss FK: {}'.format(train_loss_fk))
                print('\tValid Loss FK: {}\n'.format(valid_loss_fk))
                print("\tBest Epoch Occurred [{}/{}]".format(best_epoch, EPOCHS)) 
            
            """
            if save_option == "local":   
                torch.save(model.state_dict(), save_path+'/epoch_'+str(epoch)+'.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': train_loss,
                    }, save_path+'/currebt_epoch_model.pt')
            elif save_option == "cloud":
                torch.save(model.state_dict(), save_path+'/epoch_'+str(epoch)+'.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': train_loss,
                    }, save_path+'/current_epoch_model.pt')
                #artifact2 = wandb.Artifact(name="Model_"+robot_choice+"_" \
                #                                +model.name.replace(" ","").replace("[","_").replace("]","_").replace(",","-") \
                #                                +optimizer_choice+"_"+loss_choice+"_"+str(experiment_number+1)+'_qlim_scale_'+str(int(scale)), 
                #                            type='model')
                #artifact2.add_file(save_path+'/epoch_'+str(epoch)+'.pth')
                #run.log_artifact(artifact2)
                #torch.save(model.state_dict(), os.path.join(wandb.run.dir, "epoch_"+str(epoch)+".pth"))
            """

            # save the histories of losses
            header = ["train loss fk", "valid loss fk", "train loss ik", "valid loss ik"]
            
            df = pd.DataFrame(np.array(all_losses))
            df.to_csv(save_path+"/losses_"+robot_choice+"_"+str(dataset_samples)+".csv",
                index=False,
                header=header)
                      
            
    end_time_train = time.monotonic()
    epoch_mins, epoch_secs = epoch_time(start_time_train, end_time_train)
    
    if print_epoch:
        print('\nEnd of Training for {} - Elapsed Time: {}m {}s'.format(ik_model.name, epoch_mins, epoch_secs))    

    


    ##############################################################################################################
    # Inference
    ##############################################################################################################
    # training is done, let's run inferences and record the evaluation metrics
    print("\n\n==>Testing the trained model ...\n\n")

    if network_type == "MLP":

        ik_model = MLP(input_dim, hidden_layer_sizes, output_dim).to(device)
        fk_model = MLP(output_dim, hidden_layer_sizes, input_dim).to(device)

    elif network_type == "ResMLP":

        model = ResMLPSum(input_dim, neurons, output_dim, num_blocks).to(device)
    
    elif network_type == "DenseMLP":

        model = DenseMLP(input_dim, neurons, output_dim, num_blocks).to(device)
    
    elif network_type == "DenseMLP2":

        model = DenseMLP2(input_dim, neurons, output_dim, num_blocks).to(device)
    
    elif network_type == "DenseMLP3":

        block_config = np.zeros((1,num_blocks))   
        block_config[:,:] = layers
        block_config = block_config.squeeze(0).astype(int).tolist()
        model = DenseNet(input_dim, neurons, block_config, output_dim).to(device)





    print("==> Load FK Test Data")
    test_data_loader_fk = load_test_dataset(y_test, X_test, device)
    weights_file_fk = save_path+"/best_epoch_fk.pth"

    state_dict_fk = fk_model.state_dict()
    for n, p in torch.load(weights_file_fk, map_location=lambda storage, loc: storage).items():
        if n in state_dict_fk.keys():
            state_dict_fk[n].copy_(p)
        else:
            raise KeyError(n)


    
    # get the results from fk training     
    with torch.no_grad():
        results = forward_inference_modified(fk_model, test_data_loader_fk, fk_criterion, device, robot_choice)
    y_errors = results["y_errors_report"]
    
    #print(X_errors.shape)

    # get some inference stats
    y_errors_r = y_errors
    y_errors_r[:,:3] =y_errors_r[:,:3] * 1000
    y_errors_r[:,3:] = np.rad2deg(y_errors_r[:,3:]) 
    avg_position_error = y_errors_r[1,:3].mean()
    avg_orientation_error = y_errors_r[1,3:].mean()

    print("FK avg_position_error (mm): {}".format(avg_position_error))
    print("FK avg_orientation_error (deg): {}".format(avg_orientation_error))



    print("\n==> Load IK Test Data")
    test_data_loader_ik = load_test_dataset(X_test, y_test, device)
    weights_file_ik = save_path+"/best_epoch_ik.pth"

    state_dict_ik = ik_model.state_dict()
    for n, p in torch.load(weights_file_ik, map_location=lambda storage, loc: storage).items():
        if n in state_dict_ik.keys():
            state_dict_ik[n].copy_(p)
        else:
            raise KeyError(n)


    # get the results from training  
    with torch.no_grad():
        results = inference_modified(ik_model, test_data_loader_ik, ik_criterion, device, robot_choice)
    X_errors = results["X_errors_report"]
    
    #print(X_errors.shape)

    # get some inference stats
    X_errors_r = X_errors[:,:6]
    X_errors_r[:,:3] = X_errors_r[:,:3] * 1000
    X_errors_r[:,3:] = np.rad2deg(X_errors_r[:,3:]) 
    avg_position_error = X_errors_r[1,:3].mean()
    avg_orientation_error = X_errors_r[1,3:].mean()

    print("IK avg_position_error (mm): {}".format(avg_position_error))
    print("IK avg_orientation_error (deg): {}\n".format(avg_orientation_error))




    X_preds = results["X_preds"]
    X_desireds = results["X_desireds"]
    #X_errors_p = np.abs(X_preds - X_desireds)
    X_errors_p = results["X_errors"]
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
    if save_option == "cloud":
        inference_results = {
            "device_name": device_name,
            "data_size": dataset_samples,
            "joints_scale": scale,
            "architecture": ik_model.name,
            "network": network_type,
            "layers": layers,
            "neurons": neurons,
            "optimizer": optimizer_choice,
            "loss": loss_choice,
            "completed_epochs": epoch,
            "best_epoch": best_epoch,
            "best_valid_loss": best_valid_loss_ik,
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
        wandb.log({"inferences": inference_results_table})



    """
    inference_metrics = {
        "avg_x(mm)": X_errors_r[1,0],
        "avg_y(mm)": X_errors_r[1,1],
        "avg_z(mm)": X_errors_r[1,2],
        "avg_ro(deg)": X_errors_r[1,3],
        "avg_pi(deg)": X_errors_r[1,4],
        "avg_ya(deg)": X_errors_r[1,5]
    }

    # visualize some errors
    wandb.log({**inference_metrics})
    """
    

    #df2 = np.array(all_losses)
    #print(df2[0,0], df2[0,1])
    #test_metrics_table = wandb.Table(columns=columns)
    #test_metrics_table.add_data(df2[0,0], df2[0,1])

    if save_option == "cloud":
        wandb.finish()


            
                



