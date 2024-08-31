import numpy as np
import pandas as pd
import sys, os
from scipy import stats
from utils import *
import torch


if __name__ == '__main__':

    base_path = 'Comparative_Results_with_Numerical_Methods/'
    robot = 'RRRRRRR' # RRRRRRR, RRPRRRR
    inverse = 'SD' # MX, SVF

    for s in range(1,21):

        print('\n==> Processing joint level {}'.format(s))

        if robot == 'RRRRRRR':
            filename = 'results_7DoF-7R-Panda_m_'+inverse+'_1_seq_'+str(s)+'.csv'
        elif robot == 'RRPRRRR':
            filename = 'results_7DoF-GP66_m_'+inverse+'_1_seq_'+str(s)+'.csv'
        results = pd.read_csv(os.path.join(base_path, robot+'_Using_geometric_Jacobian', filename))
        results = pd.DataFrame(results, columns = ['D1_final','D2_final','D3_final','D4_final', 'D5_final', 'D6_final', 'D1_estimated', 'D2_estimated', 'D3_estimated', 'D4_estimated', 'D5_estimated', 'D6_estimated'])

        print(results)

        results = np.array(results)



        col1 = 3  # First column index
        col2 = 9  # Second column index
        results[:, col2] = np.where(np.sign(results[:, col1]) != np.sign(results[:, col2]),
                                            -results[:, col2], results[:, col2])
        

        col1 = 4  # First column index
        col2 = 10  # Second column index
        results[:, col2] = np.where(np.sign(results[:, col1]) != np.sign(results[:, col2]),
                                            -results[:, col2], results[:, col2])
        

        col1 = 5  # First column index
        col2 = 11  # Second column index
        results[:, col2] = np.where(np.sign(results[:, col1]) != np.sign(results[:, col2]),
                                            -results[:, col2], results[:, col2])

        X_desired = results[:,:6]
        X_estimated = results[:,6:]





        xyz_errors = np.abs(X_desired[:,:3] - X_estimated[:,:3])

        R_desireds = euler_angles_to_matrix(torch.from_numpy(X_desired[:,3:]), "XYZ")
        R_estimateds = euler_angles_to_matrix(torch.from_numpy(X_estimated[:,3:]), "XYZ")
        

        R_errors = torch.matmul(R_desireds, torch.inverse(R_estimateds))
        
        rpy_errors = matrix_to_euler_angles(R_errors, "XYZ")
        rpy_errors = torch.abs(rpy_errors)
        rpy_errors = rpy_errors.numpy() 

        X_errors = np.concatenate((xyz_errors, rpy_errors), axis=1)

        print(xyz_errors.shape)
        print(rpy_errors.shape)
        print(X_errors.shape)

        #print(R_desireds.shape)

        #sys.exit()

        X_errors[:,:3] = X_errors[:,:3] * 1000
        X_errors[:,3:] = np.rad2deg(X_errors[:,3:]) 

        """
        print()
        a = X_errors[:,3:]
        ind = np.unravel_index(np.argmax(a, axis=None), a.shape)
        print(ind)
        print(X_errors[8236,:])
        print(X_desired[8236,:])
        print(X_estimated[8236,:])
        """
        avg_position_error = X_errors[1,:3].mean()
        avg_orientation_error = X_errors[1,3:].mean()

        print("avg_position_error (mm): {}".format(avg_position_error))
        print("avg_orientation_error (deg): {}".format(avg_orientation_error))

        
        #print(X_errors_r[:,0].min())
        #print(X_errors_r[:,0].max())

        X_errors_r = np.array([[X_errors.min(axis=0)],
                                    [X_errors.mean(axis=0)],
                                    [X_errors.max(axis=0)],
                                    [X_errors.std(axis=0)]]).squeeze()
        
        #print(X_errors_report.shape)
        #sys.exit()
        
        #X_preds = results["X_preds"]
        #X_desireds = results["X_desireds"]
        #X_errors_p = np.abs(X_preds - X_desireds)
        #X_errors_p = results["X_errors"]
        #X_errors_p[:,:3] = X_errors_p[:,:3] * 1000
        #X_errors_p[:,3:] = np.rad2deg(X_errors_p[:,3:]) 
        X_percentile = stats.percentileofscore(X_errors[:,0], [1,5,10,15,20], kind='rank')
        Y_percentile = stats.percentileofscore(X_errors[:,1], [1,5,10,15,20], kind='rank')
        Z_percentile = stats.percentileofscore(X_errors[:,2], [1,5,10,15,20], kind='rank')
        Ro_percentile = stats.percentileofscore(X_errors[:,3], [1,2,3,4,5], kind='rank')
        Pi_percentile = stats.percentileofscore(X_errors[:,4], [1,2,3,4,5], kind='rank')
        Ya_percentile = stats.percentileofscore(X_errors[:,5], [1,2,3,4,5], kind='rank')


        # log this dataframe to wandb
        #columns = ["trainLoss", "validLoss"]
        #df2 = pd.DataFrame(np.array(all_losses))
    
        inference_results = {
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
        inference_results.to_csv(os.path.join(base_path, robot+'_Using_geometric_Jacobian', 'compiled_'+filename), 
                                index=False) 