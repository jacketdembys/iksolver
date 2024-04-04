from mpl_toolkits.mplot3d import Axes3D

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import random
import os
import time
from utils import *





if __name__ == "__main__":
    """
    (2) 2DoF: = Planar2D, TwoLink
    (2) 3DoF: = Planar3R, Planar2RP
    (2) 4DoF: = Cobra600 (Scara), Orion5,
    (10) 6DoF: = Jaco, Puma560, Mico, Stanford, IRB140, KR5, UR10, UR3, UR5, Puma260
    (7) 7DoF: = Panda, GP66+1, WAM, Baxter, Sawyer, Kuka LWR4+, PR2 (Right Arm)
    (1) 8DoF: = P8, 
    (1) 10DoF: = Ball, Coil 

    """

    """
    Robot type:
    C = Commensurate Robot
    I = Incommensurate Robot
    """



    # seed random generator
    torch.manual_seed(0)

    # choose robot alongside joint limits
    #robot_list = ["6DoF-Puma260", "6DoF-Puma560", "6DoF-IRB140", "6DoF-KR5", "6DoF-UR10"]
    robot_list = ["7DoF-GP66"]

    for r in robot_list:
        
        robot_choice = r
        #robot_choice = "7DoF-7R-Panda"  #"7DoF-7R-Panda, "7DoF-GP66", "8DoF-P8"
        
        if robot_choice == "6DoF-Stanford":
            
            q_lim_o = torch.tensor([[torch.deg2rad(torch.tensor(-170)), torch.deg2rad(torch.tensor(170))],
                                [torch.deg2rad(torch.tensor(-170)), torch.deg2rad(torch.tensor(170))],
                                [torch.tensor(0.3), torch.tensor(1.27)],
                                [torch.deg2rad(torch.tensor(-170)), torch.deg2rad(torch.tensor(170))],
                                [torch.deg2rad(torch.tensor(-90)), torch.deg2rad(torch.tensor(90))],
                                [torch.deg2rad(torch.tensor(-170)), torch.deg2rad(torch.tensor(170))]])

            robot_type = "I"
        
        elif robot_choice == "6DoF-Puma260":
            # https://medesign.seas.upenn.edu/index.php/Guides/PUMA260
            q_lim = torch.tensor([[torch.deg2rad(torch.tensor(-180)), torch.deg2rad(torch.tensor(110))],
                                [torch.deg2rad(torch.tensor(-75)), torch.deg2rad(torch.tensor(240))],
                                [torch.deg2rad(torch.tensor(-235)), torch.deg2rad(torch.tensor(60))],
                                [torch.deg2rad(torch.tensor(-580)), torch.deg2rad(torch.tensor(40))],
                                [torch.deg2rad(torch.tensor(-30)), torch.deg2rad(torch.tensor(200))],
                                [torch.deg2rad(torch.tensor(-215)), torch.deg2rad(torch.tensor(295))]])
        
        elif robot_choice == "6DoF-Puma560":
            
            q_lim = torch.tensor([[torch.deg2rad(torch.tensor(-160)), torch.deg2rad(torch.tensor(160))],
                                [torch.deg2rad(torch.tensor(-110)), torch.deg2rad(torch.tensor(110))],
                                [torch.deg2rad(torch.tensor(-135)), torch.deg2rad(torch.tensor(135))],
                                [torch.deg2rad(torch.tensor(-266)), torch.deg2rad(torch.tensor(266))],
                                [torch.deg2rad(torch.tensor(-100)), torch.deg2rad(torch.tensor(100))],
                                [torch.deg2rad(torch.tensor(-266)), torch.deg2rad(torch.tensor(266))]])

            robot_type = "C"
        
            
        elif robot_choice == "6DoF-Jaco":
            # https://docs.quanser.com/quarc/documentation/kinova_jaco_read_block.html
            q_lim = torch.tensor([[torch.deg2rad(torch.tensor(-360)), torch.deg2rad(torch.tensor(360))],
                                [torch.deg2rad(torch.tensor(42)), torch.deg2rad(torch.tensor(318))],
                                [torch.deg2rad(torch.tensor(17)), torch.deg2rad(torch.tensor(343))],
                                [torch.deg2rad(torch.tensor(-360)), torch.deg2rad(torch.tensor(360))],
                                [torch.deg2rad(torch.tensor(-360)), torch.deg2rad(torch.tensor(360))],
                                [torch.deg2rad(torch.tensor(-360)), torch.deg2rad(torch.tensor(360))]])

            robot_type = "C"
        
            
        elif robot_choice == "6DoF-Mico":
            # https://docs.quanser.com/quarc/documentation/kinova_6dof_mico_write_block.html
            q_lim = torch.tensor([[torch.deg2rad(torch.tensor(-10000)), torch.deg2rad(torch.tensor(10000))],
                                [torch.deg2rad(torch.tensor(50)), torch.deg2rad(torch.tensor(310))],
                                [torch.deg2rad(torch.tensor(25)), torch.deg2rad(torch.tensor(335))],
                                [torch.deg2rad(torch.tensor(-10000)), torch.deg2rad(torch.tensor(10000))],
                                [torch.deg2rad(torch.tensor(-10000)), torch.deg2rad(torch.tensor(10000))],
                                [torch.deg2rad(torch.tensor(-10000)), torch.deg2rad(torch.tensor(10000))]])

            robot_type = "C"
        
            
        elif robot_choice == "6DoF-IRB140":
            q_lim = torch.tensor([[torch.deg2rad(torch.tensor(-180)), torch.deg2rad(torch.tensor(180))],
                                [torch.deg2rad(torch.tensor(-100)), torch.deg2rad(torch.tensor(100))],
                                [torch.deg2rad(torch.tensor(-220)), torch.deg2rad(torch.tensor(60))],
                                [torch.deg2rad(torch.tensor(-200)), torch.deg2rad(torch.tensor(200))],
                                [torch.deg2rad(torch.tensor(-120)), torch.deg2rad(torch.tensor(120))],
                                [torch.deg2rad(torch.tensor(-400)), torch.deg2rad(torch.tensor(400))]])

            robot_type = "C"
        
            
        elif robot_choice == "6DoF-KR5":
            q_lim = torch.tensor([[torch.deg2rad(torch.tensor(-155)), torch.deg2rad(torch.tensor(155))],
                                [torch.deg2rad(torch.tensor(-180)), torch.deg2rad(torch.tensor(65))],
                                [torch.deg2rad(torch.tensor(-15)), torch.deg2rad(torch.tensor(158))],
                                [torch.deg2rad(torch.tensor(-350)), torch.deg2rad(torch.tensor(350))],
                                [torch.deg2rad(torch.tensor(-130)), torch.deg2rad(torch.tensor(130))],
                                [torch.deg2rad(torch.tensor(-350)), torch.deg2rad(torch.tensor(350))]])

            robot_type = "C"
        
            
        elif robot_choice == "6DoF-UR10":
            q_lim = torch.tensor([[torch.deg2rad(torch.tensor(-363)), torch.deg2rad(torch.tensor(363))],
                                [torch.deg2rad(torch.tensor(-363)), torch.deg2rad(torch.tensor(363))],
                                [torch.deg2rad(torch.tensor(-363)), torch.deg2rad(torch.tensor(363))],
                                [torch.deg2rad(torch.tensor(-363)), torch.deg2rad(torch.tensor(363))],
                                [torch.deg2rad(torch.tensor(-363)), torch.deg2rad(torch.tensor(363))],
                                [torch.deg2rad(torch.tensor(-363)), torch.deg2rad(torch.tensor(363))]])

            robot_type = "C"
        
        
        
        elif robot_choice == "7DoF-GP66":
            """
            q_lim = torch.tensor([[torch.deg2rad(torch.tensor(-170)), torch.deg2rad(torch.tensor(170))],
                                [torch.deg2rad(torch.tensor(-170)), torch.deg2rad(torch.tensor(170))],
                                #[torch.tensor(0.3), torch.tensor(1.27)],
                                [torch.tensor(-1.27), torch.tensor(-0.3)],
                                [torch.deg2rad(torch.tensor(-170)), torch.deg2rad(torch.tensor(170))],
                                [torch.deg2rad(torch.tensor(-90)), torch.deg2rad(torch.tensor(90))],
                                [torch.deg2rad(torch.tensor(-170)), torch.deg2rad(torch.tensor(170))],
                                [torch.deg2rad(torch.tensor(-170)), torch.deg2rad(torch.tensor(170))]])
            """
            
            q_lim = torch.tensor([[torch.deg2rad(torch.tensor(-90)), torch.deg2rad(torch.tensor(90))],
                                [torch.deg2rad(torch.tensor(-90)), torch.deg2rad(torch.tensor(90))],
                                [torch.tensor(-0.6), torch.tensor(0.7)],
                                #[torch.tensor(-1.27), torch.tensor(-0.3)],
                                [torch.deg2rad(torch.tensor(-90)), torch.deg2rad(torch.tensor(90))],
                                [torch.deg2rad(torch.tensor(-90)), torch.deg2rad(torch.tensor(90))],
                                [torch.deg2rad(torch.tensor(-90)), torch.deg2rad(torch.tensor(90))],
                                [torch.deg2rad(torch.tensor(-90)), torch.deg2rad(torch.tensor(90))]])

            robot_type = "I"
        
        
        elif robot_choice == "7DoF-WAM":
            
            q_lim = torch.tensor([[torch.deg2rad(torch.tensor(-170)), torch.deg2rad(torch.tensor(170))],
                                [torch.deg2rad(torch.tensor(-120)), torch.deg2rad(torch.tensor(120))],
                                [torch.deg2rad(torch.tensor(-170)), torch.deg2rad(torch.tensor(170))],
                                [torch.deg2rad(torch.tensor(-120)), torch.deg2rad(torch.tensor(120))],
                                [torch.deg2rad(torch.tensor(-170)), torch.deg2rad(torch.tensor(170))],
                                [torch.deg2rad(torch.tensor(-120)), torch.deg2rad(torch.tensor(120))],
                                [torch.deg2rad(torch.tensor(-170)), torch.deg2rad(torch.tensor(170))]])

            robot_type = "C"
        
            
        elif robot_choice == "7DoF-7R-Panda":    
            q_lim = torch.tensor([[torch.deg2rad(torch.tensor(-166)), torch.deg2rad(torch.tensor(166))],
                                    [torch.deg2rad(torch.tensor(-101)), torch.deg2rad(torch.tensor(101))],
                                    [torch.deg2rad(torch.tensor(-166)), torch.deg2rad(torch.tensor(166))],
                                    [torch.deg2rad(torch.tensor(-176)), torch.deg2rad(torch.tensor(-4))],
                                    [torch.deg2rad(torch.tensor(-166)), torch.deg2rad(torch.tensor(166))],
                                    [torch.deg2rad(torch.tensor(-1)), torch.deg2rad(torch.tensor(215))],
                                    [torch.deg2rad(torch.tensor(-166)), torch.deg2rad(torch.tensor(166))]])

            robot_type = "C"
        
            
        elif robot_choice == "7DoF-PR2-RightArm":    
            q_lim = torch.tensor([[torch.deg2rad(torch.tensor(-130)), torch.deg2rad(torch.tensor(40))],
                                    [torch.deg2rad(torch.tensor(60)), torch.deg2rad(torch.tensor(170))],
                                    [torch.deg2rad(torch.tensor(-224)), torch.deg2rad(torch.tensor(44))],
                                    [torch.deg2rad(torch.tensor(0)), torch.deg2rad(torch.tensor(133))],
                                    [torch.deg2rad(torch.tensor(-180)), torch.deg2rad(torch.tensor(180))],
                                    [torch.deg2rad(torch.tensor(0)), torch.deg2rad(torch.tensor(130))],
                                    [torch.deg2rad(torch.tensor(-180)), torch.deg2rad(torch.tensor(180))]])

            robot_type = "C"
        
            
        elif robot_choice == "8DoF-P8":    
            q_lim = torch.tensor([[torch.deg2rad(torch.tensor(-1)), torch.deg2rad(torch.tensor(1))],
                                [torch.deg2rad(torch.tensor(-1)), torch.deg2rad(torch.tensor(1))],
                                [torch.deg2rad(torch.tensor(-160)), torch.deg2rad(torch.tensor(160))],
                                [torch.deg2rad(torch.tensor(-110)), torch.deg2rad(torch.tensor(110))],
                                [torch.deg2rad(torch.tensor(-135)), torch.deg2rad(torch.tensor(135))],
                                [torch.deg2rad(torch.tensor(-266)), torch.deg2rad(torch.tensor(266))],
                                [torch.deg2rad(torch.tensor(-100)), torch.deg2rad(torch.tensor(100))],
                                [torch.deg2rad(torch.tensor(-266)), torch.deg2rad(torch.tensor(266))]])

            robot_type = "I"
        
        
        #q_lim = q_lim/scale
        scale = 10
        q_lim = q_lim*(scale/10)
        
        print("\njoint limits for scale {}:\n{}".format(scale/10, q_lim))

        t0 = torch.distributions.uniform.Uniform(q_lim[:,0], q_lim[:,1]).sample()
        DH = get_DH(robot_choice, t0)
        print(DH)

        for js in range(22,52,2):
        #for js in range(1,2):
            
            # generate dataset with a specified number of samples
            n_traj = 10000
            n_traj_steps = 100
            #n_samples = 100000000
            print_steps = 10
            data_position = []
            for s in range(n_traj):
                t0 = torch.distributions.uniform.Uniform(q_lim[:,0], q_lim[:,1]).sample()
                DH = get_DH(robot_choice, t0)
                T = forward_kinematics(DH)
                R = T[:3,:3]       
                rpy = matrix_to_euler_angles(R, "XYZ")
                #data_position.append(torch.cat([T[:3,-1], rpy, t0]).numpy())  
            
                D_current = torch.cat([T[:3,-1], rpy])
                Q_current = t0
            
                #print(t0)
                t = t0
                for i in range(n_traj_steps):
                    if robot_type == "C":
                        t += torch.deg2rad(torch.tensor(js)) # 0.5 
                    elif robot_type == "I" and robot_choice == "7DoF-GP66":
                        t[:2] += torch.deg2rad(torch.tensor(js)) # 0.5 
                        t[2] += torch.tensor(js/1000)
                        t[3:] += torch.deg2rad(torch.tensor(js)) # 0.5 
                    #print(t)
                    DH = get_DH(robot_choice, t)
                    T = forward_kinematics(DH)
                    R = T[:3,:3]       
                    rpy = matrix_to_euler_angles(R, "XYZ")
            
                    D_previous = D_current
                    Q_previous = Q_current
                    D_current = torch.cat([T[:3,-1], rpy])
                    Q_current = t
                    
                    data_position.append(torch.cat([D_previous, Q_previous, D_current, Q_current]).numpy()) 
            
            
                if s%(n_traj/print_steps) == 0:
                    print("Generated [{}] trajectories with {} samples...".format(s, i+1))
                
            data_position_a = np.array(data_position)
            
            
            # save dataset
            #if robot_choice == "6DoF-7R-Panda" or "7DoF-GP66":
            """
            header = ["x_p", "y_p", "z_p","R_p","P_p","Y_p",
                    "t1_p", "t2_p", "t3_p", "t4_p", "t5_p", "t6_p", 
                    "x_c", "y_c", "z_c","R_c","P_c","Y_c",
                    "t1_c", "t2_c", "t3_c", "t4_c", "t5_c", "t6_c"]
            """
            
            if robot_choice == "7DoF-7R-Panda" or "7DoF-GP66":
                header = ["x_p", "y_p", "z_p","R_p","P_p","Y_p",
                        "t1_p", "t2_p", "t3_p", "t4_p", "t5_p", "t6_p", "t7_p",
                        "x_c", "y_c", "z_c","R_c","P_c","Y_c",
                        "t1_c", "t2_c", "t3_c", "t4_c", "t5_c", "t6_c", "t7_c"]
            
            if robot_choice =="8DoF-P8":
                header = ["x_p", "y_p", "z_p","R_p","P_p","Y_p",
                        "t1_p", "t2_p", "t3_p", "t4_p", "t5_p", "t6_p", "t7_p", "t8_p",
                        "x_c", "y_c", "z_c","R_c","P_c","Y_c",
                        "t1_c", "t2_c", "t3_c", "t4_c", "t5_c", "t6_c", "t7_c", "t8_c"]
            
                
            df = pd.DataFrame(data_position_a)
            n_samples = data_position_a.shape[0]
            
            """
            path = "~/Documents/Research/WCCI2024/"
            df.to_csv(path+"datasets/"+robot_choice+"/data_"+robot_choice+"_"+str(n_samples)+"_qlim_scale_"+str(int(scale))+"_seq.csv",
                    index=False,
                    header=header)
            """

            
            df = pd.DataFrame(data_position_a)  # "datasets/"+robot_choice+"/
            df.to_csv("data_"+robot_choice+"_"+str(n_samples)+"_qlim_scale_"+str(int(scale))+"_seq_"+str(js)+".csv",
                    index=False,
                    header=header)
            
            
            
            data_position_a[:,:3] = data_position_a[:,:3] * 1000
            
            print("Data generation done for {}! - {} samples with {} joint step\n".format(robot_choice, n_samples, js))
