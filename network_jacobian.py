import torch 
import torch.nn as nn
import time
from utils import *




class MLP(nn.Module):
    def __init__(self, input_dim, h_sizes, output_dim):
        super().__init__()

        self.name = "MLP [{}, {}, {}]".format(str(input_dim), str(h_sizes).replace("[","").replace("]",""), str(output_dim))
        self.input_dim = input_dim
        self.h_sizes = h_sizes
        self.output_dim = output_dim
        
        self.input_fc = nn.Linear(self.input_dim, self.h_sizes[0])
        self.relu_activation = nn.ReLU()
        
        self.hidden_fc = nn.ModuleList()
        for i in range(len(self.h_sizes)-1):
            self.hidden_fc.append(nn.Linear(self.h_sizes[i], self.h_sizes[i+1]))
        
        self.output_fc = nn.Linear(self.h_sizes[len(self.h_sizes)-1], self.output_dim)

        
        
    def forward(self, x):

        x = self.input_fc(x)
        x = self.relu_activation(x)  

        for i in range(len(self.h_sizes)-1):
            x = self.hidden_fc[i](x)
            x = self.relu_activation(x)

        x = self.output_fc(x)
        x_temp = x

        return x, x_temp

    def get_network_jacobian(self, inputs, output_poses, device):

            # compute the Jacobian
            batch = inputs.shape[0]
            input_size = inputs.shape[1]
            output_size = self.output_dim #* self.dim_position
            

            # initialize a tensor to hold the Jacobian
            J = torch.zeros(batch, 1 , input_size, output_size)
            #print('J: ', J.shape)
            #print('output_size: ', output_size)
            #print('output_poses: ', output_poses.shape)
            #print('inputs: ', inputs.shape)

            t = time.time()
            for j in range(output_size):
                g = torch.autograd.grad(output_poses[:,j], 
                                        inputs, 
                                        grad_outputs=torch.ones_like(output_poses[:,j]).to(device),
                                        retain_graph=True)
                g = g[0].permute(1,0)
                g = torch.reshape(g, (batch, 1, input_size))
                J[:,:,:,j] = g
                #print('g{}: {}'.format(j, g))


            #print('g: ', g.shape)
            #print('J: ', J.shape)

            J_reshape = torch.reshape(J, (batch, -1, self.input_dim))
            #print(J[0,:,:,0])
            #print(J[0,:,:,1])
            #print(J_reshape[0,:,:])
            #print('J_reshape: ', J_reshape.shape)

            J_reshape = J_reshape.permute(0, 2, 1) 
            #print('J_reshape: ', J_reshape.shape)
            #print(J_reshape[0,:,:])
            
            return J_reshape
    
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
        
        #print(theta)
        #print(d)
        #print(a)
        #print(alpha)
        

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


class FKLoss(nn.Module):
    def __init__(self, robot_choice, device):
        #super(FKLoss, self).__init__()
        super().__init__()
        self.criterion = nn.MSELoss(reduction="mean")
        #self.criterion = nn.L1Loss(reduction="mean")
        self.robot_choice = robot_choice

    def forward(self, joints, poses):
        #inputs_fk = torch.zeros_like(targets)
        joints_fk = torch.clone(poses)
        joints_fk.retain_grad()

        
        DH = get_DH_2(self.robot_choice)
        T_successive, T_total = joint_angle_to_transformation_matrix(joints, DH, device)
        R = T_total[:,:3,:3]
        rpy = matrix_to_euler_angles(R, "XYZ")
        joints_fk = torch.cat([T_total[:,:3,-1], rpy[:,:]], axis=1)
        
        loss = self.criterion(joints_fk, poses)
        
        return loss

if __name__ == '__main__':

    # set the device to compute autograd for the network jacobian
    device = torch.device('cuda:0')
    
    # intializa model
    seed_number = 0
    torch.manual_seed(seed_number)
    torch.cuda.manual_seed(seed_number)
    torch.backends.cudnn.deterministic = True


    
    input_size = 6
    hidden_sizes = [64,64] 
    output_size = 7
    model = MLP(input_size,
                hidden_sizes,
                output_size).to(device)
    
    # test model prediction
    input = torch.randn(100000,6).to(device)
    input.requires_grad = True
    output, _ = model(input)
    print(output)

    # test network jacobian function
    #J_net = model.get_network_jacobian(input, output, device)
    #print(J_net.shape)

    # test FKloss
    stime = time.time()
    robot_choice = "7DoF-7R-Panda"
    criterion = FKLoss(robot_choice, device)
    loss = criterion(output, input)
    etime = time.time()
    print(etime-stime)
    print(loss)