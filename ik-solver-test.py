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
import matplotlib.pyplot as plt
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




robot_choice = "7DoF-7R-Panda"
dataset_samples = 1000000
scale = 10
n_DoF = 7 
input_dim = 6+6+7 #6
output_dim = 7
batch_size = 100000
dataset_type = "seq"
device = torch.device('cuda:0') # torch.device('cpu') 
network_type = "DenseMLP"
layers = 12
num_blocks = 5
neurons = 1024
hidden_layer_sizes = np.zeros((1,layers))    
hidden_layer_sizes[:,:] = neurons
hidden_layer_sizes = hidden_layer_sizes.squeeze(0).astype(int).tolist()


# seed the random generators
seed_choice = True
if seed_choice: 
    seed_number = 0
    random.seed(seed_number)
    np.random.seed(seed_number)
    torch.manual_seed(seed_number)
    torch.cuda.manual_seed(seed_number)
    torch.backends.cudnn.deterministic = True


data = pd.read_csv('/home/retina/dembysj/Dropbox/WCCI2024/docker/datasets/'+robot_choice+'/data_'+robot_choice+'_'+str(int(dataset_samples))+'_qlim_scale_'+str(int(scale))+'_seq.csv')
#data = pd.read_csv('/home/retina/dembysj/Dropbox/WCCI2024/docker/datasets/7DoF-7R-Panda/data_7DoF-7R-Panda_1000000_qlim_scale_10_seq.csv')
data_a = np.array(data)
train_data_loader, test_data_loader, X_validate, y_validate, X_train, y_train, X_test, y_test, sc_in = load_dataset(data_a, n_DoF, batch_size, robot_choice, dataset_type, device)
save_path = "results/7DoF-7R-Panda/keep_results/DenseMLP_7DoF-7R-Panda_blocks_5_neurons_1024_batch_100000_Adam_lq_1_qlim_scale_10_samples_1000000"

criterion = nn.MSELoss(reduction="mean")

print("\n\n==>Testing the trained model ...\n\n")
weights_file = save_path+"/best_epoch.pth"
if network_type == "MLP":
    model = MLP(input_dim, hidden_layer_sizes, output_dim).to(device)
    #model = MLP(mapping_size*2, hidden_layer_sizes, output_dim).to(device)
elif network_type == "ResMLP":
    model = ResMLPSum(input_dim, neurons, output_dim, num_blocks).to(device)
elif network_type == "DenseMLP":
    model = DenseMLP(input_dim, neurons, output_dim, num_blocks).to(device)
elif network_type == "DenseMLP2":
    model = DenseMLP2(input_dim, neurons, output_dim, num_blocks).to(device)

state_dict = model.state_dict()
for n, p in torch.load(weights_file, map_location=lambda storage, loc: storage).items():
    if n in state_dict.keys():
        state_dict[n].copy_(p)
    else:
        raise KeyError(n)

# get the results from testing  
"""
test_data_loader = load_test_dataset(X_test, y_test, device)
with torch.no_grad():
    results = inference_modified(model, test_data_loader, criterion, device, robot_choice)
X_errors = results["X_errors_report"]

#print(X_errors.shape)

# get some inference stats
X_errors_r = X_errors[:,:6]
X_errors_r[:,:3] = X_errors_r[:,:3] * 1000
X_errors_r[:,3:] = np.rad2deg(X_errors_r[:,3:]) 
avg_position_error = X_errors_r[1,:3].mean()
avg_orientation_error = X_errors_r[1,3:].mean()

print("avg_position_error (mm): {}".format(avg_position_error))
print("avg_orientation_error (deg): {}".format(avg_orientation_error))
"""

traj_1 = data_a[0:100,:]
traj_2 = data_a[100:200,:]
traj_3 = data_a[600:700,:]
traj_4 = data_a[1100:1200,:]
#traj_5 = data_a[2700:2800,:]
#traj_5 = data_a[3600:3700,:]
data_a = np.concatenate((traj_1,
                         traj_2,
                         traj_3,
                         traj_4), axis=0)

#vis_traj = 3
#X_test = data_a[:vis_traj*100,:19]
#y_test = data_a[:vis_traj*100,19:]
X_test = data_a[:,:19]
y_test = data_a[:,19:]
test_data_loader = load_test_dataset_2(X_test, y_test, device, sc_in)
with torch.no_grad():
    results = inference_modified(model, test_data_loader, criterion, device, robot_choice)
X_errors = results["X_errors_report"]
X_preds = results["X_preds"]


#print(X_errors.shape)

# get some inference stats
X_errors_r = X_errors[:,:6]
X_errors_r[:,:3] = X_errors_r[:,:3] * 1000
X_errors_r[:,3:] = np.rad2deg(X_errors_r[:,3:]) 
avg_position_error = X_errors_r[1,:3].mean()
avg_orientation_error = X_errors_r[1,3:].mean()

print("avg_position_error (mm): {}".format(avg_position_error))
print("avg_orientation_error (deg): {}".format(avg_orientation_error))




# Visualize a trajectory and predict it
data_a[:,:3] = data_a[:,:3]*1000
X_preds[:,:3] = X_preds[:,:3]*1000

### Visualize 4DoF Dataset
fig = plt.figure(figsize=(10, 10))
ax = plt.axes(projection='3d')
#ax.scatter(data_a[:vis_traj*100,0], data_a[:vis_traj*100,1], data_a[:vis_traj*100,2], s=10, c='b', marker='.')
#ax.scatter(X_preds[:vis_traj*100,0], X_preds[:vis_traj*100,1], X_preds[:vis_traj*100,2], s=10, c='g', marker='.')
ax.scatter(data_a[:,0], data_a[:,1], data_a[:,2], s=10, c='b') #, marker='.'
ax.scatter(X_preds[:,0], X_preds[:,1], X_preds[:,2], s=10, c='g')
ax.scatter(0,0,0,s=20, marker='o', c='r')
ax.legend(["desired pose","predicted pose","robot base"])
ax.set(xlabel=r'$X (mm)$', ylabel=r'$Y (mm)$', zlabel=r'$Z (mm)$') #, title='Visualization of '+robot_choice+ ' dataset')
#ax.view_init(60, 25)
ax.view_init(50, 25)
plt.savefig("test_trajectories_summary.pdf", format="pdf", bbox_inches="tight")
plt.show()