# libraries
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
import os
import sys
import argparse

from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import decomposition
from sklearn import manifold
from tqdm import tqdm
from scipy import stats

# get DH matrix based on robot choice
def get_DH(robot_choice, t):
    # columns: t, d, a, alpha
    if robot_choice == "2DoF-2R":
        DH = torch.tensor([[t[0], 0, 1, 0],
                           [t[1], 0, 1, 0]])
    elif robot_choice == "3DoF-3R":
        DH = torch.tensor([[t[0], 0, 1, 0],
                           [t[1], 0, 1, torch.pi/2],
                           [t[2], 0, 0, 0]])
    elif robot_choice == "3DoF-3R-2":
        DH = torch.tensor([[t[0], 380/1000, 0, 0],
                           [t[1],        0, 280/1000, -torch.pi/2],
                           [t[2],        0, 280/1000, 0]])
    elif robot_choice == "4DoF-2RPR":
        DH = torch.tensor([[t[0], 400/1000, 250/1000, 0],
                           [t[1],        0, 150/1000, torch.pi],
                           [   0,     t[2],        0, 0],
                           [t[3], 150/1000,        0, 0]])
    elif robot_choice == "6DoF-6R-Puma260":
        DH = torch.tensor([[t[0],           0,          0,        -torch.pi/2],
                           [t[1],  125.4/1000, 203.2/1000,                  0],
                           [t[2],           0,  -7.9/1000,         torch.pi/2],
                           [t[3],  203.2/1000,          0,        -torch.pi/2],
                           [t[4],           0,          0,         torch.pi/2],
                           [t[5],   63.5/1000,          0,                  0]])
    elif robot_choice == "7DoF-7R-Panda":
        DH = torch.tensor([[t[0],    0.333,      0.0,           0],
                           [t[1],      0.0,      0.0, -torch.pi/2],
                           [t[2],    0.316,      0.0,  torch.pi/2],
                           [t[3],      0.0,   0.0825,  torch.pi/2],
                           [t[4],    0.384,  -0.0825, -torch.pi/2],
                           [t[5],      0.0,      0.0,  torch.pi/2],
                           [t[6],    0.107,    0.088,  torch.pi/2]])
    return DH





# A matrix
def A_matrix(t,d,a,al):
    # the inputs of torch.sin and torch.cos are expressed in rad
    A = torch.tensor([[torch.cos(t), -torch.sin(t)*torch.cos(al),  torch.sin(t)*torch.sin(al), a*torch.cos(t)],
                      [torch.sin(t),  torch.cos(t)*torch.cos(al), -torch.cos(t)*torch.sin(al), a*torch.sin(t)],
                      [           0,               torch.sin(al),               torch.cos(al),              d],
                      [           0,                           0,                           0,              1]])
    return A

# Forward Kinematics
def forward_kinematics(DH):

    n_DoF = DH.shape[0]
    T = torch.eye(4,4)
    for i in range(n_DoF):
        A = A_matrix(*DH[i,:])
        T = torch.matmul(T, A)
    
    return T

# weight initialization
def weights_init_uniform_rule(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # get the number of the inputs
        n = m.in_features
        y = 1.0/np.sqrt(n)
        m.weight.data.uniform_(-y, y)
        m.bias.data.fill_(0)


def weights_init_normal_rule(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # get the number of the inputs
        n = m.in_features
        y = 1.0/np.sqrt(n)
        m.weight.data.normal_(0.0,y)
        m.bias.data.fill_(0)


def weights_init_xavier_uniform_rule(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # get the number of the inputs
        n = m.in_features
        y = 1.0/np.sqrt(n)        
        nn.init.xavier_normal_(m.weight.data, gain=nn.init.calculate_gain('relu'))
        m.bias.data.fill_(0)

def weights_init_xavier_normal_rule(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # get the number of the inputs
        n = m.in_features
        y = 1.0/np.sqrt(n)
        nn.init.xavier_normal_(m.weight.data)
        m.bias.data.fill_(0)


def weights_init_kaiming_uniform_rule(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # get the number of the inputs
        n = m.in_features
        y = 1.0/np.sqrt(n)
        nn.init.kaiming_uniform_(m.weight.data)
        m.bias.data.fill_(0)

def weights_init_kaiming_normal_rule(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # get the number of the inputs
        n = m.in_features
        y = 1.0/np.sqrt(n)
        nn.init.kaiming_normal_(m.weight.data)
        m.bias.data.fill_(0)




# classes and functions
## Conversion from PyTorch 3D GitHub repository
def _angle_from_tan(
    axis: str, other_axis: str, data, horizontal: bool, tait_bryan: bool
) -> torch.Tensor:
    """
    Extract the first or third Euler angle from the two members of
    the matrix which are positive constant times its sine and cosine.

    Args:
        axis: Axis label "X" or "Y or "Z" for the angle we are finding.
        other_axis: Axis label "X" or "Y or "Z" for the middle axis in the
            convention.
        data: Rotation matrices as tensor of shape (..., 3, 3).
        horizontal: Whether we are looking for the angle for the third axis,
            which means the relevant entries are in the same row of the
            rotation matrix. If not, they are in the same column.
        tait_bryan: Whether the first and third axes in the convention differ.

    Returns:
        Euler Angles in radians for each matrix in data as a tensor
        of shape (...).
    """

    i1, i2 = {"X": (2, 1), "Y": (0, 2), "Z": (1, 0)}[axis]
    if horizontal:
        i2, i1 = i1, i2
    even = (axis + other_axis) in ["XY", "YZ", "ZX"]
    if horizontal == even:
        return torch.atan2(data[..., i1], data[..., i2])
    if tait_bryan:
        return torch.atan2(-data[..., i2], data[..., i1])
    return torch.atan2(data[..., i2], -data[..., i1])

def _index_from_letter(letter: str) -> int:
    if letter == "X":
        return 0
    if letter == "Y":
        return 1
    if letter == "Z":
        return 2
    raise ValueError("letter must be either X, Y or Z.")

def matrix_to_euler_angles(matrix: torch.Tensor, convention: str) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to Euler angles in radians.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
        convention: Convention string of three uppercase letters.

    Returns:
        Euler angles in radians as tensor of shape (..., 3).
    """
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")
    i0 = _index_from_letter(convention[0])
    i2 = _index_from_letter(convention[2])
    tait_bryan = i0 != i2
    if tait_bryan:
        central_angle = torch.asin(
            matrix[..., i0, i2] * (-1.0 if i0 - i2 in [-1, 2] else 1.0)
        )
    else:
        central_angle = torch.acos(matrix[..., i0, i0])

    o = (
        _angle_from_tan(
            convention[0], convention[1], matrix[..., i2], False, tait_bryan
        ),
        central_angle,
        _angle_from_tan(
            convention[2], convention[1], matrix[..., i0, :], True, tait_bryan
        ),
    )
    return torch.stack(o, -1)



# vanilla MLP architecture
class MLP_2(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.name = "MLP[]"
        self.input_fc = nn.Linear(input_dim, 250)
        self.hidden_fc = nn.Linear(250, 100)
        self.output_fc = nn.Linear(100, output_dim)

    def forward(self, x):

        # x = [batch size, height, width]
        batch_size = x.shape[0]
        
        x = x.view(batch_size, -1)
        # x = [batch size, height * width]
        
        h_1 = F.relu(self.input_fc(x))
        # h_1 = [batch size, 250]

        h_2 = F.relu(self.hidden_fc(h_1))
        # h_2 = [batch size, 100]

        y_pred = self.output_fc(h_2)
        # y_pred = [batch size, output dim]

        return y_pred, h_2

class MLP(nn.Module):
    def __init__(self, input_dim, h_sizes, output_dim):
        super().__init__()

        self.name = "MLP [{}, {}, {}]".format(str(input_dim), str(h_sizes).replace("[","").replace("]",""), str(output_dim))
        self.input_dim = input_dim
        self.h_sizes = h_sizes
        self.output_dim = output_dim
        
        self.input_fc = nn.Linear(self.input_dim, self.h_sizes[0])
        
        self.hidden_fc = nn.ModuleList()
        for i in range(len(self.h_sizes)-1):
            self.hidden_fc.append(nn.Linear(self.h_sizes[i], self.h_sizes[i+1]))
        
        self.output_fc = nn.Linear(self.h_sizes[len(self.h_sizes)-1], self.output_dim)

        self.selu_activation = nn.SELU()
        self.relu_activation = nn.ReLU()
        self.prelu_activation = nn.PReLU()
        self.lrelu_activation = nn.LeakyReLU()
        self.sigmoid_activation = nn.Sigmoid()
        self.batch_norm_fc = nn.BatchNorm1d(20000)

    def forward(self, x):

        x = self.input_fc(x)
        #x = self.batch_norm_fc(x)
        x = self.relu_activation(x)  # ReLU(), Sigmoid(), LeakyReLU(negative_slope=0.1)

        for i in range(len(self.h_sizes)-1):
            x = self.hidden_fc[i](x)
            #x = self.batch_norm_fc(x)
            x = self.relu_activation(x)

        x = self.output_fc(x)
        x_temp = x

        return x, x_temp 


class ResMLP_2(nn.Module):
    def __init__(self, input_dim, h_sizes, output_dim):
        super().__init__()

        self.name = "ResMLP [{}, {}, {}]".format(str(input_dim), str(h_sizes).replace("[","").replace("]",""), str(output_dim))
        self.input_dim = input_dim
        self.h_sizes = h_sizes
        self.output_dim = output_dim
        
        self.input_fc = nn.Linear(self.input_dim, self.h_sizes[0])
        self.hidden_fc_1 = nn.Linear(self.h_sizes[0], self.h_sizes[1])
        self.hidden_fc_2 = nn.Linear(self.h_sizes[1], self.h_sizes[2])
        self.hidden_fc_3 = nn.Linear(self.h_sizes[2], self.h_sizes[3])
        self.output_fc = nn.Linear(self.h_sizes[len(self.h_sizes)-1], self.output_dim)       

        self.selu_activation = nn.SELU()
        self.relu_activation = nn.ReLU()
        self.prelu_activation = nn.PReLU()
        self.lrelu_activation = nn.LeakyReLU()
        self.sigmoid_activation = nn.Sigmoid()
        self.batch_norm_fc = nn.BatchNorm1d(20000)

    def forward(self, x):

        x = self.input_fc(x)
        x = self.relu_activation(x)  # ReLU(), Sigmoid(), LeakyReLU(negative_slope=0.1)

        h1 = self.hidden_fc_1(x)
        h1 = self.relu_activation(h1)

        h2 = self.hidden_fc_2(h1)
        h2 = self.relu_activation(h2)

        h3 = self.hidden_fc_3(h2+h1)
        h3 = self.relu_activation(h3)

        o = self.output_fc(h3+h2+h1)
        x_temp = o

        return o, x_temp 


class DenseMLP(nn.Module):
    def __init__(self, input_dim, h_sizes, output_dim):
        super().__init__()

        self.name = "DenseMLP [{}, {}, {}]".format(str(input_dim), str(h_sizes).replace("[","").replace("]",""), str(output_dim))
        self.input_dim = input_dim
        self.h_sizes = h_sizes
        self.output_dim = output_dim
        
        self.input_fc = nn.Linear(self.input_dim, self.h_sizes[0])
        self.hidden_fc_1 = nn.Linear(self.h_sizes[0], self.h_sizes[1])
        self.hidden_fc_2 = nn.Linear(self.h_sizes[1], self.h_sizes[2])
        self.hidden_fc_3 = nn.Linear(self.h_sizes[2]*2, self.h_sizes[3])
        self.output_fc = nn.Linear(self.h_sizes[len(self.h_sizes)-1]*3, self.output_dim)       

        self.selu_activation = nn.SELU()
        self.relu_activation = nn.ReLU()
        self.prelu_activation = nn.PReLU()
        self.lrelu_activation = nn.LeakyReLU()
        self.sigmoid_activation = nn.Sigmoid()
        self.batch_norm_fc = nn.BatchNorm1d(20000)

    def forward(self, x):

        x = self.input_fc(x)
        x = self.relu_activation(x)  # ReLU(), Sigmoid(), LeakyReLU(negative_slope=0.1)

        h1 = self.hidden_fc_1(x)
        h1 = self.relu_activation(h1)

        h2 = self.hidden_fc_2(h1)
        h2 = self.relu_activation(h2)

        #print(h2.shape)
        #sys.exit(0)

        h3 = self.hidden_fc_3(torch.cat((h2,h1),1))
        h3 = self.relu_activation(h3)

        o = self.output_fc(torch.cat((h3,h2,h1),1))
        x_temp = o

        return o, x_temp 


# count network parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# data loader
class LoadIKDataset(Dataset):
    def __init__(self, inputs_array, outputs_array, device):
        x_temp = inputs_array
        y_temp = outputs_array

        self.x_data = torch.tensor(x_temp, dtype=torch.float32) #.to(device) 
        self.y_data = torch.tensor(y_temp, dtype=torch.float32) #.to(device)  

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        current_input = self.x_data[idx, :]
        current_output = self.y_data[idx, :]
        
        sample = {'input': current_input,
                  'output': current_output}
        return sample
    

    def __len__(self):
        return len(self.x_data)


# function to load the dataset
def load_dataset(data, n_DoF, batch_size, robot_choice, dataset_type, device):

    # file data_4DoF
    #X = data[:,:3]
    #y = data[:,6:]

    # file data_4DOF_2
    if robot_choice == "6DoF-6R-Puma260":
        X = data[:,:6]
        y = data[:,6:]
    if robot_choice == "7DoF-7R-Panda":
        if dataset_type == "seq":
            print("==> Sequence dataset ...")
            X = data[:,:19]
            y = data[:,19:]
        elif dataset_type == "1_to_1": 
            print("==> 1 to 1 dataset ...")
            X = data[:,:6]
            y = data[:,6:] #13]
    if robot_choice == "3DoF-3R":
        if dataset_type == "seq":
            print("==> Sequence dataset ...")
            X = data[:,:7]
            y = data[:,7:]
        elif dataset_type == "1_to_1": 
            print("==> 1 to 1 dataset ...")
            X = data[:,:2]
            y = data[:,2:5] #13]

        
    #y = data[:,:2]
    #X = data[:,2:]
        
    # split in train and test sets
    
    X_train, X_validate, y_train, y_validate = train_test_split(X, 
                                                                y, 
                                                                test_size = 0.1,
                                                                random_state = 1)

    X_train, X_test, y_train, y_test = train_test_split(X_train, 
                                                        y_train, 
                                                        test_size = 0.1,
                                                        random_state = 1)
    
    
    """
    n_samples = len(data[:,0])
    X_train = X[:int(0.8*n_samples),:]
    X_validate = X[int(0.8*n_samples):int(0.9*n_samples),:]
    X_test = X[int(0.9*n_samples):,:]

    y_train = y[:int(0.8*n_samples),:]
    y_validate = y[int(0.8*n_samples):int(0.9*n_samples),:]
    y_test = y[int(0.9*n_samples):,:]
    """
    
    
    sc_in = MinMaxScaler(copy=True, feature_range=(0, 1))
    sc_out = MinMaxScaler(copy=True, feature_range=(0, 1))
    
    X_train = sc_in.fit_transform(X_train)
    X_validate = sc_in.transform(X_validate) 
    X_test = sc_in.transform(X_test) 
    

    """
    min_value = np.min(X_train)
    max_value = np.max.max(X_train)
    range = max_value - min_value
    X_train = (X_train - min_value) / range
    X_validate = (X_validate - min_value) / range
    X_test = (X_test - min_value) / range

    print(X_train.min(), X_train.max())
    print(X_validate.min(), X_validate.max())
    print(X_test.min(), X_test.max())
    """

    
    
    #xx = torch.from_numpy(X_train)
    #xx = xx
    #print(xx)
    #print(B.to(torch.float64))
    #X_train = input_mapping(torch.from_numpy(X_train),B.to(torch.float64))
    #X_test = input_mapping(torch.from_numpy(X_test),B.to(torch.float64))
    #X_train = X_train.numpy()
    #X_test = X_test.numpy()
    
    #X_train = X_train_i
    #X_test = X_test_i

    #y_train = sc_out.fit_transform(y_train)
    #y_test = sc_out.transform(y_test) 

    print("==> Shape X_train: ", X_train.shape)
    print("==> Shape y_train: ", y_train.shape)

    train_data = LoadIKDataset(X_train, y_train, device)
    test_data = LoadIKDataset(X_validate, y_validate, device)

    train_data_loader = DataLoader(dataset=train_data,
                                   batch_size=batch_size,
                                   shuffle=False,
                                   drop_last=True,
                                   pin_memory=False,
                                   num_workers=8,
                                   persistent_workers=True)

    test_data_loader = DataLoader(dataset=test_data,
                                   batch_size=batch_size,
                                   drop_last=False,
                                   shuffle=False,
                                   pin_memory=False,
                                   num_workers=8,
                                   persistent_workers=True)

    return train_data_loader, test_data_loader, X_validate, y_validate, X_train, y_train, X_test, y_test




# function to load the dataset
def load_dataset_dist(data, n_DoF, batch_size, robot_choice, dataset_type, device):

    # file data_4DoF
    #X = data[:,:3]
    #y = data[:,6:]

    # file data_4DOF_2
    if robot_choice == "6DoF-6R-Puma260":
        X = data[:,:6]
        y = data[:,6:]
    if robot_choice == "7DoF-7R-Panda":
        if dataset_type == "seq":
            print("==> Sequence dataset ...")
            X = data[:,:19]
            y = data[:,19:]
        elif dataset_type == "1_to_1": 
            print("==> 1 to 1 dataset ...")
            X = data[:,:6]
            y = data[:,6:] #13]
    if robot_choice == "3DoF-3R":
        if dataset_type == "seq":
            print("==> Sequence dataset ...")
            X = data[:,:7]
            y = data[:,7:]
        elif dataset_type == "1_to_1": 
            print("==> 1 to 1 dataset ...")
            X = data[:,:2]
            y = data[:,2:5] #13]


    # split in train and test sets    
    X_train_priv, X_validate_priv, y_train, y_validate = train_test_split(X, 
                                                                        y, 
                                                                        test_size = 0.1,
                                                                        random_state = 1)

    X_train_priv, X_test_priv, y_train, y_test = train_test_split(X_train_priv, 
                                                                y_train, 
                                                                test_size = 0.1,
                                                                random_state = 1)
        
    
    sc_in = MinMaxScaler(copy=True, feature_range=(0, 1))
    sc_out = MinMaxScaler(copy=True, feature_range=(0, 1))
    
    X_train_priv = sc_in.fit_transform(X_train_priv)
    X_validate_priv = sc_in.transform(X_validate_priv) 
    X_test_priv = sc_in.transform(X_test_priv) 
    
    print("==> Shape X_train: ", X_train_priv.shape)
    print("==> Shape y_train: ", y_train.shape)

    train_data_priv = LoadIKDataset(X_train_priv, y_train, device)
    test_data_priv = LoadIKDataset(X_validate_priv, y_validate, device)

    train_data_loader_priv = DataLoader(dataset=train_data_priv,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   drop_last=True,
                                   pin_memory=False,
                                   num_workers=8,
                                   persistent_workers=True)

    test_data_loader_priv = DataLoader(dataset=test_data_priv,
                                   batch_size=batch_size,
                                   drop_last=False,
                                   shuffle=False,
                                   pin_memory=False,
                                   num_workers=8,
                                   persistent_workers=True)
    


    X_train_dist = X_train_priv[:,13:]
    X_validate_dist = X_validate_priv[:,13:]
    X_test_dist = X_test_priv[:,13:]
    train_data_dist = LoadIKDataset(X_train_dist, y_train, device)
    test_data_dist = LoadIKDataset(X_validate_dist, y_validate, device)

    train_data_loader_dist = DataLoader(dataset=train_data_dist,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   drop_last=True,
                                   pin_memory=False,
                                   num_workers=8,
                                   persistent_workers=True)

    test_data_loader_dist = DataLoader(dataset=test_data_dist,
                                   batch_size=batch_size,
                                   drop_last=False,
                                   shuffle=False,
                                   pin_memory=False,
                                   num_workers=8,
                                   persistent_workers=True)
    


    output = {
        "train_data_loader_priv": train_data_loader_priv,
        "test_data_loader_priv": test_data_loader_priv,
        "X_validate_priv": X_validate_priv,
        "y_validate": y_validate,
        "X_train_priv": X_train_priv,
        "y_train": y_train,
        "X_test_priv": X_test_priv,
        "y_test": y_test,
        "train_data_loader_dist": train_data_loader_dist,
        "test_data_loader_dist": test_data_loader_dist,
        "X_validate_dist": X_validate_dist,
        "X_train_dist": X_train_dist,
        "X_test_dist": X_test_dist
    }


    return output




# function to load the dataset
def load_test_dataset(X_test, y_test, device):

    print("==> Shape X_test: ", X_test.shape)
    print("==> Shape y_test: ", y_test.shape)

    test_data = LoadIKDataset(X_test, y_test, device)

    test_data_loader = DataLoader(dataset=test_data,
                                   batch_size=1,
                                   drop_last=False,
                                   shuffle=False,
                                   pin_memory=False,
                                   num_workers=8,
                                   persistent_workers=True)

    return test_data_loader




# train function
def train(model, iterator, optimizer, criterion, criterion_type, batch_size, device, epoch, EPOCHS, scheduler, scaler):
    epoch_loss = 0
    epoch_loss_2 = 0
    model.train()
    i = 0



    if criterion_type == "ld":
        criterion_2 = nn.MSELoss(reduction="mean")

    #B_dict = {}
    #B_dict['basic'] = torch.eye(32,3)
    #B = B_dict['basic'].to(device)
    
    #with tqdm(total=(len(iterator) - len(iterator) % batch_size)) as t:
    with tqdm(total=len(iterator), desc='Epoch: [{}/{}]'.format(epoch+1, EPOCHS), disable=True) as t:
        for data in iterator:
        #for data in tqdm(iterator, desc="Training", leave=False):
            optimizer.zero_grad()
            x, y = data['input'], data['output']
            x.requires_grad = True

            x = x.to(device)
            y = y.to(device)
            
            #x = input_mapping(x,B)
            
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                y_pred, _ = model(x)

                if criterion_type == "ld":
                    #loss = criterion(y_pred, x)
                    wp = 1
                    wq = 0
                    loss = wp*criterion(y_pred, y)+wq*criterion_2(y_pred, y)
                else:
                    loss = criterion(y_pred, y)

            #print("y_pred\n:", y_pred)

            #print(x.shape)
            #print(y.shape)
            #print(y_pred.shape)
            #sys.exit()

            #print("\nTrain Epoch {} at batch {}".format(epoch, i))
            """
            if i == 1:
                print("\nTrain Epoch {} at batch {}".format(epoch, i))
                print(y_pred[:5,:])
                print(y[:5,:])
                #sys.exit()
            """
            
            # optimizer.zero_grad()
            #loss = criterion(y_pred, y)

            #print(x)
            #print(y_pred) 
            

            #make_dot(loss, params=dict(list(model.named_parameters()))).render("loss", format="png")
            
            #loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            #optimizer.step()
            #scheduler.step()


            scaler.scale(loss).backward()
            scaler.step(optimizer)
            #scheduler.step()
            scaler.update()

            epoch_loss += loss.item()
            #if criterion_type == "ld":
                #epoch_loss_2 += loss2.item()
            t.set_postfix_str('Train loss: {:.6f}'.format(epoch_loss/len(iterator)))
            t.update()

            i += 1

            #sys.exit()
    
    
    
    #print("Total batches {}".format(i))
        #if criterion_type == "ld":
            #print('\n\tTrain FK Loss: {}'.format(epoch_loss/len(iterator)))
            #print('\tTrain L2 Loss: {}'.format(epoch_loss_2/len(iterator)))
    return epoch_loss/len(iterator)




def train_back(model, iterator, optimizer, criterion, criterion_type, batch_size, device, epoch, EPOCHS, scheduler, scaler):
    epoch_loss = 0
    epoch_loss_2 = 0
    model.train()
    i = 0



    if criterion_type == "ld":
        criterion_2 = nn.MSELoss(reduction="mean")

    #B_dict = {}
    #B_dict['basic'] = torch.eye(32,3)
    #B = B_dict['basic'].to(device)
    
    #with tqdm(total=(len(iterator) - len(iterator) % batch_size)) as t:
    with tqdm(total=len(iterator), desc='Epoch: [{}/{}]'.format(epoch+1, EPOCHS), disable=True) as t:
        for data in iterator:
        #for data in tqdm(iterator, desc="Training", leave=False):
            optimizer.zero_grad()
            x, y = data['input'], data['output']
            x.requires_grad = True

            x = x.to(device)
            y = y.to(device)
            
            #x = input_mapping(x,B)
            
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                y_pred, _ = model(x)

                if criterion_type == "ld":
                    #loss = criterion(y_pred, x)
                    wp = 1
                    wq = 0
                    loss = wp*criterion(y_pred, y)+wq*criterion_2(y_pred, y)
                else:
                    loss = criterion(y_pred, y)

            #print("y_pred\n:", y_pred)

            #print(x.shape)
            #print(y.shape)
            #print(y_pred.shape)
            #sys.exit()

            #print("\nTrain Epoch {} at batch {}".format(epoch, i))
            """
            if i == 1:
                print("\nTrain Epoch {} at batch {}".format(epoch, i))
                print(y_pred[:5,:])
                print(y[:5,:])
                #sys.exit()
            """
            
            # optimizer.zero_grad()
            #loss = criterion(y_pred, y)

            #print(x)
            #print(y_pred) 
            

            #make_dot(loss, params=dict(list(model.named_parameters()))).render("loss", format="png")
            
            #loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            #optimizer.step()
            #scheduler.step()


            scaler.scale(loss).backward()
            scaler.step(optimizer)
            #scheduler.step()
            scaler.update()

            epoch_loss += loss.item()
            #if criterion_type == "ld":
                #epoch_loss_2 += loss2.item()
            t.set_postfix_str('Train loss: {:.6f}'.format(epoch_loss/len(iterator)))
            t.update()

            i += 1

            #sys.exit()
    
    
    
    #print("Total batches {}".format(i))
        #if criterion_type == "ld":
            #print('\n\tTrain FK Loss: {}'.format(epoch_loss/len(iterator)))
            #print('\tTrain L2 Loss: {}'.format(epoch_loss_2/len(iterator)))
    return epoch_loss/len(iterator)


def train_dist(model_priv, model_dist, iterator_priv, iterator_dist, optimizer_dist, criterion, criterion_type, batch_size, device, epoch, EPOCHS, scheduler, scaler, alpha):
    epoch_loss = 0
    epoch_loss_2 = 0
    model_priv.eval()
    model_dist.train()
    i = 0
    



    if criterion_type == "ld":
        criterion_2 = nn.MSELoss(reduction="mean")

    #B_dict = {}
    #B_dict['basic'] = torch.eye(32,3)
    #B = B_dict['basic'].to(device)

    
    #with tqdm(total=(len(iterator) - len(iterator) % batch_size)) as t:
    with tqdm(total=len(iterator_dist), desc='Epoch: [{}/{}]'.format(epoch+1, EPOCHS), disable=True) as t:
        for (data_dist, data_priv) in zip (iterator_dist, iterator_priv):
        #for data in tqdm(iterator, desc="Training", leave=False):
            optimizer_dist.zero_grad()

            # for distillation phase
            x_dist, y_dist = data_dist['input'], data_dist['output']
            x_dist.requires_grad = True
            x_dist = x_dist.to(device)
            y_dist = y_dist.to(device)


            # from privilege phase
            x_priv, y_priv = data_priv['input'], data_priv['output']
            x_priv = x_priv.to(device)
            y_priv = y_priv.to(device)

            #print(x_dist[0,:],y_dist[0,:])
            #print(x_priv[0,:],y_priv[0,:])
            #sys.exit()
            
            with torch.no_grad():
                y_pred_priv, _ = model_priv(x_priv)
            
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                y_pred_dist, _ = model_dist(x_dist)

                loss_dist = criterion(y_pred_dist, y_dist)
                loss_imit = criterion(y_pred_dist, y_pred_priv)
                loss = (1-alpha)*loss_dist + alpha*loss_imit           
           

            #make_dot(loss, params=dict(list(model.named_parameters()))).render("loss", format="png")
            
            #loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            #optimizer.step()
            #scheduler.step()


            scaler.scale(loss).backward()
            scaler.step(optimizer_dist)
            #scheduler.step()
            scaler.update()

            epoch_loss += loss.item()
            #if criterion_type == "ld":
                #epoch_loss_2 += loss2.item()
            t.set_postfix_str('Train loss: {:.6f}'.format(epoch_loss/len(iterator_dist)))
            t.update()

            i += 1

            #sys.exit()
    
    
    
    #print("Total batches {}".format(i))
        #if criterion_type == "ld":
            #print('\n\tTrain FK Loss: {}'.format(epoch_loss/len(iterator)))
            #print('\tTrain L2 Loss: {}'.format(epoch_loss_2/len(iterator)))
    return epoch_loss/len(iterator_dist)

# evaluation function 
def evaluate(model, iterator, criterion, criterion_type, device, epoch, EPOCHS):
    epoch_loss = 0
    model.eval()

    if criterion_type == "ld":
        criterion_2 = nn.MSELoss(reduction="mean")

    #B_dict = {}
    #B_dict['basic'] = torch.eye(32,3)
    #B = B_dict['basic'].to(device)
    
    with torch.no_grad():
        #for data in tqdm(iterator, desc="Evaluating", leave=False):        
        with tqdm(total=len(iterator), desc='Epoch: [{}/{}]'.format(epoch+1, EPOCHS), disable=True) as t:
            for data in iterator:
                x = data['input'].to(device)
                y = data['output'].to(device)
                #x.requires_grad = True

                #x = input_mapping(x,B)
                
                y_pred, _ = model(x)
                #loss = criterion(y_pred, y)
                #loss = criterion(y_pred, x)  
                
                if criterion_type == "ld":
                    #loss = criterion(y_pred, x)
                    wp = 1
                    wq = 0
                    loss = wp*criterion(y_pred, y)+wq*criterion_2(y_pred, y)
                else:
                    loss = criterion(y_pred, y)
                
                epoch_loss += loss.item()
    
                t.set_postfix_str('Valid loss: {:.6f}'.format(epoch_loss/len(iterator)))
                t.update()

    return epoch_loss/len(iterator)

# make predictions
def inference(model, iterator, criterion, device, robot_choice):
    model.eval()
    y_preds = []
    y_desireds = []
    X_desireds = []
    
    for data in iterator:
        x = data['input'].to(device)
        y = data['output'].to(device)

        #x = input_mapping(x,B)
        
        y_pred, _ = model(x)
        y_preds.append(y_pred.detach().cpu().numpy().squeeze())
        y_desireds.append(y.detach().cpu().numpy().squeeze())
        #X_desireds.append(x.detach().cpu().numpy().squeeze())


    y_desireds = np.array(y_desireds)
    #X_desireds = np.array(X_desireds)
    X_desireds = reconstruct_pose(y_desireds, robot_choice)
    y_preds = np.array(y_preds)
    X_preds = reconstruct_pose(y_preds, robot_choice)

    #print(X_preds.shape)
    #print(X_desireds.shape)
    
    X_errors = np.abs(X_preds - X_desireds)
    y_errors = np.abs(y_preds - y_desireds)

    #print(X_errors.shape)

    X_errors_report = np.array([[X_errors.min(axis=0)],
                                [X_errors.mean(axis=0)],
                                [X_errors.max(axis=0)],
                                [X_errors.std(axis=0)]]).squeeze()
    
    results = {
        "y_preds": y_preds,
        "X_preds": X_preds,
        "y_desireds": y_desireds,
        "X_desireds": X_desireds,
        "X_errors": X_errors_report
    }
    return results


def inference_modified(model, iterator, criterion, device, robot_choice):
    model.eval()
    y_preds = []
    y_desireds = []
    X_desireds = []
    
    for data in iterator:
        x = data['input'].to(device)
        y = data['output'].to(device)

        #x = input_mapping(x,B)
        
        y_pred, _ = model(x)
        y_preds.append(y_pred.detach().cpu().numpy().squeeze())
        y_desireds.append(y.detach().cpu().numpy().squeeze())
        #X_desireds.append(x.detach().cpu().numpy().squeeze())


    y_desireds = np.array(y_desireds)
    y_preds = np.array(y_preds)
    #X_desireds = np.array(X_desireds)
    X_desireds, X_preds, X_errors = reconstruct_pose_modified(y_desireds, y_preds, robot_choice)
    
    X_errors_report = np.array([[X_errors.min(axis=0)],
                                [X_errors.mean(axis=0)],
                                [X_errors.max(axis=0)],
                                [X_errors.std(axis=0)]]).squeeze()
    
    results = {
        "y_preds": y_preds,
        "X_preds": X_preds,
        "y_desireds": y_desireds,
        "X_desireds": X_desireds,
        "X_errors": X_errors,
        "X_errors_report": X_errors_report
    }
    return results

def inference_FK(model, iterator, criterion, device):
    model.eval()
    y_preds = []
    y_desireds = []
    X_desireds = []
    for data in iterator:
        x = data['input'].to(device)
        y = data['output'].to(device)
        y_pred, _ = model(x)
        y_preds.append(y_pred.detach().cpu().numpy().squeeze())
        y_desireds.append(y.detach().cpu().numpy().squeeze())
        X_desireds.append(x.detach().cpu().numpy().squeeze())

    y_desireds = np.array(y_desireds)
    X_desireds = np.array(X_desireds)
    #X_desireds = reconstruct_pose(y_desireds, robot_choice)
    y_preds = np.array(y_preds)
    #X_preds = reconstruct_pose(y_preds, robot_choice)

    #X_errors = np.abs(X_preds - X_desireds)
    y_errors = np.abs(y_preds - y_desireds)

    y_errors_report = np.array([[y_errors.min(axis=0)],
                                [y_errors.mean(axis=0)],
                                [y_errors.max(dim=0)]]).squeeze()
    
    results = {
        "y_preds": y_preds,
        #"X_preds": X_preds,
        "y_desireds": y_desireds,
        #"X_desireds": X_desireds,
        "y_errors": y_errors_report
    }
    return results

# reconstruct positions in cartesian space from predictions
def reconstruct_pose(y_preds, robot_choice):
    y_preds = torch.from_numpy(y_preds)
    n_samples = y_preds.shape[0]
    pose = []
    for i in range(n_samples):
        t = y_preds[i,:]
        DH = get_DH(robot_choice, t)
        T = forward_kinematics(DH)
        if robot_choice == "4DoF-2RPR":
            # x,y,t1,t2,t3 where x,y (m) and t (rad)
            pose.append(T[:3,-1].numpy())
        
        elif robot_choice == "6DoF-6R-Puma260":
            R = T[:3,:3] 
            rpy = matrix_to_euler_angles(R, "XYZ")
            # x,y,z,R,P,Y,t1,t2,t3,t4,t5,t6 where x,y,z (m) and t (rad)
            #print(T[:3,-1])
            #print(rpy)
            pose.append(torch.cat([T[:3,-1], rpy, t]).numpy())
        
        elif robot_choice == "7DoF-7R-Panda":
            R = T[:3,:3] 
            rpy = matrix_to_euler_angles(R, "XYZ")
            # x,y,z,R,P,Y,t1,t2,t3,t4,t5,t6,t7 where x,y,z (m) and t (rad)
            #print(T[:3,-1])
            #print(rpy)
            pose.append(torch.cat([T[:3,-1], rpy, t]).numpy())

    X_pred = np.array(pose)
    return X_pred


def reconstruct_pose_modified(y_desireds, y_preds, robot_choice):
    y_desireds = torch.from_numpy(y_desireds)
    y_preds = torch.from_numpy(y_preds)
    n_samples = y_preds.shape[0]

    pose_desireds = []
    pose_preds = []
    pose_errors = []
    
    for i in range(n_samples):

        # set the joints
        t_desireds = y_desireds[i,:]
        t_preds = y_preds[i,:]

        # compute the forward kinematics
        DH_desireds = get_DH(robot_choice, t_desireds)
        T_desireds = forward_kinematics(DH_desireds)

        DH_preds = get_DH(robot_choice, t_preds)
        T_preds = forward_kinematics(DH_preds)       
        
        if robot_choice == "7DoF-7R-Panda":
            R_desireds = T_desireds[:3,:3]
            R_preds = T_preds[:3,:3] 

            rpy_desireds = matrix_to_euler_angles(R_desireds, "XYZ")
            rpy_preds = matrix_to_euler_angles(R_preds, "XYZ")

            R_errors = torch.matmul(R_desireds, torch.inverse(R_preds))
            #T_errors = torch.matmul(T_desireds, torch.inverse(T_preds))
            #R_errors = T_errors[:3,:3] 
            rpy_errors = matrix_to_euler_angles(R_errors, "XYZ")   
            rpy_errors = torch.abs(rpy_errors)         
            #position_errors = T_errors[:3,-1]
            position_errors = torch.abs(T_desireds[:3,-1]-T_preds[:3,-1])
            
            # x,y,z,R,P,Y,t1,t2,t3,t4,t5,t6,t7 where x,y,z (m) and t (rad)
            #print(T[:3,-1])
            #print(rpy)
            pose_desireds.append(torch.cat([T_desireds[:3,-1], rpy_desireds, t_desireds]).numpy())
            pose_preds.append(torch.cat([T_preds[:3,-1], rpy_preds, t_preds]).numpy())
            pose_errors.append(torch.cat([position_errors, rpy_errors]).numpy())



    X_desireds = np.array(pose_desireds)
    X_preds = np.array(pose_preds)
    X_errors = np.array(pose_errors)
    return X_desireds, X_preds, X_errors
    

# compute epoch time
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


# def FFT embedding from this paper: Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains
def input_mapping(x, B):
    if B is None:
        return x
    else:
        x_proj = torch.matmul((2*torch.pi*x), B.T)
        #print(x.shape)
        #print(B.shape)
        #print(x_proj.shape)
        #print(torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1).shape)
    
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        

class FourierLayer(nn.Module):
    def __init__(self, in_features, out_features, scale):
        super().__init__()
        B = torch.randn(in_features, out_features)*scale
        self.register_buffer("B", B)

    def forward(self, x):
        x_proj = torch.matmul(2*math.pi*x, self.B)
        out = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        return out

class FourierMLP(nn.Module):
    def __init__(self, input_dim, fourier_dim, h_sizes, output_dim, scale):
        super().__init__()
        
        self.name = "FourierMLP [{}, {}, {}]".format(str(input_dim), str(h_sizes).replace("[","").replace("]",""), str(output_dim))
        self.input_dim = input_dim
        self.fourier_dim = fourier_dim
        self.h_sizes = h_sizes
        self.output_dim = output_dim

        self.fourier_fc = FourierLayer(self.input_dim, self.fourier_dim, scale)
        self.input_fc = nn.Linear(2*self.fourier_dim, self.h_sizes[0])
        
        self.hidden_fc = nn.ModuleList()
        for i in range(len(self.h_sizes)-1):
            self.hidden_fc.append(nn.Linear(self.h_sizes[i], self.h_sizes[i+1]))
        
        self.output_fc = nn.Linear(self.h_sizes[len(self.h_sizes)-1], self.output_dim)

        self.selu_activation = nn.SELU()
        self.relu_activation = nn.ReLU()
        self.prelu_activation = nn.PReLU()
        self.lrelu_activation = nn.LeakyReLU()
        self.sigmoid_activation = nn.Sigmoid()
        self.batch_norm_fc = nn.BatchNorm1d(20000)

    def forward(self, x):

        x = self.fourier_fc(x)
        x = self.input_fc(x)
        x = self.relu_activation(x)  # ReLU(), Sigmoid(), LeakyReLU(negative_slope=0.1)

        for i in range(len(self.h_sizes)-1):
            x = self.hidden_fc[i](x)
            #x = self.batch_norm_fc(x)
            x = self.relu_activation(x)

        x = self.output_fc(x)
        x_temp = x

        return x, x_temp 
        




def get_DH_2(robot_choice):
    # columns: t, d, a, alpha

    if robot_choice == "7DoF-7R-Panda":
        DH = torch.tensor([[0,    0.333,      0.0,           0],
                            [0,      0.0,      0.0, -torch.pi/2],
                            [0,    0.316,      0.0,  torch.pi/2],
                            [0,      0.0,   0.0825,  torch.pi/2],
                            [0,    0.384,  -0.0825, -torch.pi/2],
                            [0,      0.0,      0.0,  torch.pi/2],
                            [0,    0.107,    0.088,  torch.pi/2]])

    return DH



def joint_angle_to_transformation_matrix(theta_ndh, DH, device):
        
        #print("theta.shape: {}".format(theta_ndh.shape))
        #print(theta.shape[0])
        #print(theta.shape[1])

        
        #print("theta: {}".format(theta_ndh))
        #print("theta: {}".format(theta_ndh[:,:,0]))
       
        batch = theta_ndh.shape[0]
        joint_number = theta_ndh.shape[1]

        # populate the DH with the thetas and have as many as the batch size
        #print("DH.shape: {}".format(self.DH.shape))
        DH = DH.to(device)
        DH = DH.repeat(batch, 1).view(batch, joint_number, 4)
        DH[:,:,0] = theta_ndh
        #DH[:,2,2] = theta_ndh[:,0,2]
        #DH[:,:2,3] = theta_ndh[:,0,:2]
        #print("DH.shape: {}".format(DH.shape))
        #print("DH.shape: {}".format(DH))
        
        #theta = theta_ndh.clone()
        #print("theta.shape 2", theta.shape)

        #print(DH)
        theta = DH[:,:,0]
        d = DH[:,:,1]
        a = DH[:,:,2]
        alpha = DH[:,:,3]
        
        #print("theta: {}".format(theta))
        #print("d: {}".format(d))
        #print("alpha: {}".format(alpha))
        #print("a: {}".format(a))

        theta = theta.view(-1,1)
        d = d.view(-1, 1)        
        a = a.view(-1, 1)
        alpha = alpha.view(-1, 1)
        
        #print("theta:\n",theta)
        #print("d:\n",d)
        #print("a:\n",a)
        #print("alpha:\n",alpha)
        

        row_1 = torch.cat( (torch.cos(theta), -torch.sin(theta)*torch.cos(alpha),  torch.sin(theta)*torch.sin(alpha), a*torch.cos(theta)), 1 )    
        row_2 = torch.cat( (torch.sin(theta),  torch.cos(theta)*torch.cos(alpha), -torch.cos(theta)*torch.sin(alpha), a*torch.sin(theta)), 1 )   
            
        zeros = torch.autograd.Variable(torch.zeros(joint_number,1).to(device))
        zeros = zeros.repeat(batch,1).view(-1, 1)         
        ones = torch.autograd.Variable(torch.ones(joint_number,1).to(device))
        ones = ones.repeat(batch,1).view(-1, 1)

        #print(joint_number)
        #print(zeros.shape)
        #print(alpha.shape)
        #print(d.shape)
        
        row_3 = torch.cat( (zeros, torch.sin(alpha), torch.cos(alpha), d), 1 )
        row_4 = torch.cat( (zeros, zeros, zeros, ones), 1 )
        T_successive = torch.cat((row_1, row_2, row_3, row_4), 1).view(batch, joint_number, 4, 4)  

        T_total = T_successive[:,0,:,:].view(batch,1,4,4)
        #print("T_successive.shape): {}".format(T_successive.shape))
        #print("T_total.shape): {}".format(T_total.shape))      

        for i in range(1, joint_number):
            temp_total_transformation = torch.matmul(T_total, T_successive[:,i,:,:].view(batch,1,4,4))
            T_total = temp_total_transformation    

        return T_successive, T_total.view(batch,4,4)


# compute loss function by employing the FK 
class FKLoss(nn.Module):
    def __init__(self, robot_choice, device):
        #super(FKLoss, self).__init__()
        super().__init__()
        self.criterion = nn.MSELoss(reduction="mean")
        #self.criterion = nn.L1Loss(reduction="mean")
        self.robot_choice = robot_choice
        self.device = device

    def forward(self, joints_pred, joints_des):
        #inputs_fk = torch.zeros_like(targets)
        #joints_fk = torch.clone(poses)
        #joints_fk.retain_grad()

        
        DH = get_DH_2(self.robot_choice)
        T_successive_pred, T_total_pred = joint_angle_to_transformation_matrix(joints_pred, DH, self.device)
        R_pred = T_total_pred[:,:3,:3]
        rpy_pred = matrix_to_euler_angles(R_pred, "XYZ")
        pose_pred = torch.cat([T_total_pred[:,:3,-1], rpy_pred[:,:]], axis=1)


        T_successive_des, T_total_des = joint_angle_to_transformation_matrix(joints_des, DH, self.device)
        R_des = T_total_des[:,:3,:3]
        rpy_des = matrix_to_euler_angles(R_des, "XYZ")
        pose_des = torch.cat([T_total_des[:,:3,-1], rpy_des[:,:]], axis=1)

        #print()
        #print("T_total:")
        #print(T_total)
        #print("Joints_fk:")
        #print(pose_pred)
        #print("poses")
        #print(pose_des)
        #print()
        
        loss = self.criterion(pose_pred, pose_des)
        
        return loss
    

class FKLossB(nn.Module):
    def __init__(self, robot_choice):
        #super(FKLoss, self).__init__()
        super().__init__()
        self.criterion = nn.MSELoss(reduction="mean")
        #self.criterion = nn.L1Loss(reduction="mean")
        self.robot_choice = robot_choice

    def forward(self, joints, poses):
        #inputs_fk = torch.zeros_like(targets)
        joints_fk = torch.clone(poses)
        joints_fk.retain_grad()


        #DH = torch.vmap(get_DH, in)(self.robot_choice, joints)
        #print(DH.shape)

        #sys.exit()



        #print(targets)
        #print(inputs_fk)
        #sys.exit()
        for i in range(joints.shape[0]):
            #print()
            t = joints[i,:]
            DH = get_DH(self.robot_choice, t)
            #print(DH)
            T = forward_kinematics(DH)
            #print(T.type)
            if self.robot_choice == "7DoF-7R-Panda":
                R = T[:3,:3]
                rpy = matrix_to_euler_angles(R, "XYZ")
                
                #inputs_fk[i,:] = T[:3,-1]   
                joints_fk[i,:] = torch.cat([T[:3,-1], rpy])

        #inputs_fk = inputs_fk
        #print(joints_fk)
        #print(poses)
        #print("here")
        #sys.exit()
        loss = self.criterion(joints_fk, poses)
        #print(loss)
        return loss
