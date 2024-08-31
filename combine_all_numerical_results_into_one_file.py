import pandas as pd
import os
from tqdm import tqdm

if __name__ == '__main__':
    
    # List of CSV filenames (assuming they are in the same directory)
    base_path = 'Comparative_Results_with_Numerical_Methods/'
    robot = 'RRRRRRR'
    inverse = 'SD'
    if robot == 'RRRRRRR':
        robot_call = '7DoF-7R-Panda' 
    elif robot == 'RRPRRRR':
        robot_call = '7DoF-GP66'
    base_path = os.path.join(os.path.join(base_path, robot+'_Using_geometric_Jacobian'))
    csv_files = [base_path+f'/compiled_results_'+robot_call+'_m_'+inverse+'_1_seq_'+str(i)+'.csv' for i in range(1,21)]

    # Initialize an empty list to store the DataFrames
    df_list = []

    # Loop through the file names and read each CSV
    for file in tqdm(csv_files):
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file)
        
        # Append the DataFrame to the list
        df_list.append(df)

    # Concatenate all DataFrames into a single DataFrame
    combined_df = pd.concat(df_list, ignore_index=True)

    print('\n',combined_df)

    # Optionally, you can save the combined DataFrame to a new CSV file
    combined_df.to_csv(os.path.join(base_path, 'combined_results_'+robot_call+'_m_'+inverse+'_1_seq.csv'), index=False)

    print("CSV files have been successfully combined to: {}.".format(base_path))