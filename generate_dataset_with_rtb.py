import roboticstoolbox as rtb
import numpy as np
import sys
import pandas as pd
import time
from utils import *

# intitialize the robot
robot_choice = "Panda"
if robot_choice == "Panda":
    panda = rtb.models.DH.Panda()
    print(panda)
    #panda.links[2].a = 100
    #print(panda
    #print(panda.qlim)

# seed the random generator
np.random.seed(532023)                                  

# set the number of samples
n_samples = 100000
print_steps = n_samples/10
data_list = []

start = time.time()
for i in range(n_samples):

    # sample a joint configuration 
    q = np.random.uniform(panda.qlim[0,:], panda.qlim[1,:])

    # compute the forward kinematics
    T = panda.fkine(q)

    # get the position and orientation components
    t, rpy = T.t, T.rpy()

    # find the jacobian matrix
    J = panda.jacobe(q=q)

    # find the hessian matrix 
    #H = panda.hessiane(q=q, Je=J)
    
    # store the generated dataset
    #data_list.append(np.concatenate((t, rpy, q, J.flatten(), H.flatten())))
    data_list.append(np.concatenate((t, rpy, q, J.flatten())))

    # report on the evolution
    if i%print_steps == 0:
        print("|=> current number of generated samples: {}".format(i))

end = time.time()


mins, secs = epoch_time(start, end)
    

print('\nElapsed time: {}m {}s'.format(mins, secs)) 

# convert in numpy array
data_array = np.array(data_list)

# save to csv file
df = pd.DataFrame(data_array)
df.to_csv("data_"+robot_choice+"_"+str(n_samples)+".csv",
          index=False,
          header=False)






