%% prepare workspace
clc, clear, close all

%% Add path to implementation of different inverses and datasets
addpath('../docker/datasets/7DoF-7R-Panda-Steps/')
addpath('../docker/datasets/7DoF-GP66-Steps/')
addpath(genpath('Numerical_Methods/'))


%% initialize global parameters and running modes
robot_list = ["RRPRRRR"] %, "RRPRRRR"];

for r=1:length(robot_list)

    robot = robot_list(r); %'RRRRRRR';   % RRRRRRR (7R-Panda), RRPRRRR (2RP4R-GP66+1)
    units  = ["m"];
    inverses = ["SD"] %, "SD", "SVF"]; % SD, SVF
    jacobian_type = 'geometric';                % "numerical", "geometric", "analytical"
    motion = strcat('Comparative_Results_with_Numerical_Methods/',robot,'_Using_',jacobian_type,'_Jacobian');
    
    mode_run = "run";                           % "run" or "debug"
    mode_print = "False";
    mode_print_result = "True";
    mode_save = "True";
    mode_save_path = "False";
    total_computation = 0;                      % total computation time
    
    %% create a folder to store the results
    % motion is the name of the directory when the result are stored
    if ~exist(motion, 'dir')
        mkdir(motion)
    end
    
    for seq=1:20
    
        %% load the related sample points and initialize the summary matrices   
        if (strcmp(robot, 'RRRRRRR'))
            robot_call = '7DoF-7R-Panda';
            native_data = readtable(strcat('review_data_',robot_call,'_1000000_qlim_scale_10_seq_',num2str(seq),'_test.csv'));
            native_data = table2array(native_data);
            dataPoints7DoFR = native_data(:,20:26);            % load poses (positions + orientations) 14-19 for poses / 20-26 for joints
            dataPoints7DoFR_algo = native_data(:,7:13);       % load joint configurations
            samples = length(dataPoints7DoFR);
            summaryTable = zeros(samples,33);
        elseif (strcmp(robot, 'RRPRRRR'))
            robot_call = '7DoF-GP66';
            native_data = readtable(strcat('review_data_',robot_call,'_1000000_qlim_scale_10_seq_',num2str(seq),'_test.csv'));
            native_data = table2array(native_data);
            dataPoints7DoFR = native_data(:,20:26);            % load poses (positions + orientations)
            dataPoints7DoFR_algo = native_data(:,7:13);       % load joint configurations
            samples = length(dataPoints7DoFR);
            summaryTable = zeros(samples,33);
        end
        
        
        %% choose the number of samples and iterations for "debug" or "run" modes
        if strcmp(mode_run, "debug")
            samples = 1;
            iterations = 1000;
        else
            iterations = 1000;
        end
        
        
        %% loop through the inverses
        for j1=1:length(inverses)
        
            %% loop through the units
            for i1=1:length(units)
                
                %% loop through the choice of alpha
                indexes = [1];   % experiments done with 1
                %indexes = [0.03; 0.06; 0.09; 0.1; 0.2; 0.4; 0.6; 0.8; 1];
                %indexes = [0.03; 0.5; 1];
                %indexes = [0.1; 0.01];   %, 0.1, 0.01];
                
                for j = 1:length(indexes)
                
                    % set the unit, inverse, and alpha choices
                    str_unit_chosen = units(i1);
                    inverse_chosen = inverses(j1);
                    alpha = indexes(j);          
        
                    fprintf("\nRobot [%s]", robot_call);
                    fprintf("\nInverse [%s]", inverse_chosen);
                    fprintf("\nUnit [%s]", str_unit_chosen);
                    fprintf("\nAlpha [%d]", alpha);
        
                    %% choose the unit to investigate
                    if(str_unit_chosen == "m")                     % choose the m
                        unit_chosen = 1; 
                        unit_applied = 1000;
                    elseif (str_unit_chosen == "dm")               % choose the dm
                        unit_chosen = 10;
                        unit_applied = 100;
                    elseif (str_unit_chosen == "cm")               % choose the cm
                        unit_chosen = 100;
                        unit_applied = 10;
                    elseif (str_unit_chosen == "mm")               % choose the mm
                        unit_chosen = 1000;
                        unit_applied = 1;
                    end   
                    
                    if strcmp(mode_save_path, "True")
                        diary(char(strcat(motion, '/', inverse_chosen, '_', str_unit_chosen,'_', num2str(alpha), '.txt')));
                    end
        
                    %% set IK path information
                    for s=1:samples


                        fprintf("\nData Point [%d]\n", s);
                        
                        if strcmp(mode_run, "debug")                    
                            if  strcmp(robot, 'RRRRRRR')  
                                s = 1;  
                                Q_initial = dataPoints7DoFR_algo(s,:)';  % Home Pose (Joint Configuration)                        
                                Q_final_d = dataPoints7DoFR(s,:)';       % Example Pose (Joint Configuration)
                                Q_final_d(3) = Q_final_d(3)*unit_chosen;
                                dim = 6;      
                            elseif  strcmp(robot, 'RRPRRRR')  
                                s = 1;   
                                Q_initial = dataPoints7DoFR_algo(s,:)';  
                                Q_initial(3) = Q_initial(3)*unit_chosen;                         
                                Q_final_d = dataPoints7DoFR(s,:)'; 
                                Q_final_d(3) = Q_final_d(3)*unit_chosen;                         
                                dim = 6;
                            end
        
                        elseif strcmp(mode_run, "run")
        
                            if  strcmp(robot, 'RRRRRRR')                     
                                Q_initial = dataPoints7DoFR_algo(s,:)';                        
                                Q_final_d = dataPoints7DoFR(s,:)'; 
                                Q_final_d(3) = Q_final_d(3)*unit_chosen;
                                dim = 6;                   
                            elseif  strcmp(robot, 'RRPRRRR')  
                                Q_initial = dataPoints7DoFR_algo(s,:)';  
                                Q_initial(3) = Q_initial(3)*unit_chosen;                         
                                Q_final_d = dataPoints7DoFR(s,:)'; 
                                Q_final_d(3) = Q_final_d(3)*unit_chosen;                         
                                dim = 6; 
                            end
        
                        end
        
                        
        
                        %% (combined) IK pose accepted tolerance
                        maximum_iteration       = iterations;
                        current_iteration       = 0;
                        D_estimated             = zeros(dim, 1);
                        
                        position_error          = 0.001 * unit_chosen;          % 1 mm
                        orientation_error       = 0.0175;                       % 0.0175 rad = 1 deg, 0.001 = 0.5 deg                        
                        emax                    = 0.02;
                        
        
        
                        %% DH and initial pose               
                        DH = getDH_rad(robot, Q_initial, unit_chosen);
                        pc_robot_configuration = getRobotConfiguration(robot, unit_chosen, DH);
                        T = fkine(pc_robot_configuration, Q_initial);
                        D_initial = getPose_rad(T, dim);
                        Q_current = Q_initial;
                        D_current = D_initial;
                        D_current_p = D_initial;
                                                        
                        % uncomment this if starting with a final joint configuration instead
                        
                        DH = getDH_rad(robot, Q_final_d, unit_chosen);
                        pc_robot_configuration = getRobotConfiguration(robot, unit_chosen, DH);
                        T_final = fkine(pc_robot_configuration, Q_final_d);  
                        D_final = getPose_rad(T_final, dim);
                        
                        %D_final = Q_final_d;
        
                        %D_initial
                        %D_final
                        %error('Stopped here on purpose for debugging ...')
                        
                        if strcmp(mode_save_path, "True")
                            saveToFile(D_current, motion, str_unit_chosen, inverse_chosen, alpha, "w");
                        end
                     
                        
                        %% get the errors for the inverses
                        if strcmp(robot, "RRP")
                            e = D_final - D_current;
        
                        else
                            
                            if strcmp(jacobian_type, "geometric")
                                ex = D_final(1:3) - D_current(1:3);
                                eo = 0.5*(cross(T(1:3,1), T_final(1:3,1)) + ...
                                          cross(T(1:3,2), T_final(1:3,2)) + ...
                                          cross(T(1:3,3), T_final(1:3,3)));
                                e = [ex; eo];  
                            elseif strcmp(jacobian_type, "numerical")    
                                e = D_final - D_current;
                            end
                        end
        
                        %% variables for distance calculation
                        ideal_distance = 0;
                        performed_distance = 0;
                        final = 0;
                        
                        %fprintf("\nBeginning the iterative process");
                        tic
        
                        %% loop to look for the solution
                        while(final == 0)
        
                            %% compute the traveled distances
                            performed_distance = getDistance(D_current, D_current_p, robot, performed_distance);                
                            D_current_p = D_current;
        
                            
                            %% DH and current pose
                            %DH = getDH_rad(robot, Q_current, unit_chosen);
                            %pc_robot_configuration = getRobotConfiguration(robot, unit_chosen, DH);
                            %T = fkine(pc_robot_configuration, Q_current);
                            %D_current = getPose_rad(T, length(D_final));
        
        
                            %% get the errors for the inverses
                            %{
                            ex = D_final(1:3) - D_current(1:3);
                            eo = 0.5*(cross(T(1:3,1), T_final(1:3,1)) + ...
                                      cross(T(1:3,2), T_final(1:3,2)) + ...
                                      cross(T(1:3,3), T_final(1:3,3)));
                            e = [ex; eo];
                            %}
                            
                            % display the current estimated joint and pose vector
                            if strcmp(mode_print, "True")
                                displayJoints(Q_current, robot);
                                displayPose(D_current, length(D_final));
                            end
        
                            %% get the Jacobian                        
                            J = getJacobianMatrix(DH, D_final, D_current, Q_current, robot, jacobian_type, unit_chosen);
            
                            %{
                            if strcmp(robot, 'RRVPRVRVPRR') 
                                J(:,3) = zeros(6,1);
                                J(:,6) = zeros(6,1);
                                J(:,8) = zeros(6,1);
                            end
                            %}
                            
                            
                            if strcmp(mode_print, "True")
                                % Display the current jacobian
                                fprintf("\nCurrent Jacobian:")
                                J  %#ok<NOPTS>
                                %J = round(J, 4)
                            end
                            
                            %% check if a singularity happened
                            ss = checkIfSingularityHappened(J, Q_current, current_iteration);
                                                
                            if strcmp(inverse_chosen, "SD")
                                %% get d_D and d_Q
                                gamma_max = 0.5;
                                d_Q = select_dampinv2(J, gamma_max, alpha*e) %#ok<NOPTS>
                                d_Q = round(d_Q, 4) %#ok<NOPTS>
                            else
                                
                                %% get the inverse Jacobian
                                inv_J = getInverseJacobian(J, inverse_chosen, e, robot);
                                d_D = alpha*e;
                                
                                %% get d_D and d_Q
                                d_Q = inv_J*d_D;
                                
                                if strcmp(mode_print, "True")
                                    fprintf("\nInverse Jacobian:")
                                    inv_J  %#ok<NOPTS>
                                    %inv_J = round(inv_J, 6)
        
                                    fprintf("\nd_D - current difference in pose:")
                                    d_D %#ok<NOPTS>
        
                                    fprintf("\nd_Q - current difference in joint:")
                                    d_Q %#ok<NOPTS>
                                    %d_Q = wrapToPi(d_Q)%#ok<NOPTS>
                                    d_Q = round(d_Q, 4)%#ok<NOPTS>
                                end
                                
                            end
        
                            %% get the current (new) joint configuration Q_current
                            Q_current = Q_current + d_Q; 
                            %Q_current = wrapToPi(Q_current);
        
                            %% estimate the current combined pose error
                            DH = getDH_rad(robot, Q_current, unit_chosen);
                            pc_robot_configuration = getRobotConfiguration(robot, unit_chosen, DH);
                            T = fkine(pc_robot_configuration, Q_current);
                            D_current = getPose_rad(T, dim);               
        
                            
                            %% get the errors for the inverses
                            if strcmp(robot, "RRP")
                                e = D_final - D_current;
                            
                            else
                                
                                if strcmp(jacobian_type, "geometric")
                                    ex = D_final(1:3) - D_current(1:3);
                                    eo = 0.5*(cross(T(1:3,1), T_final(1:3,1)) + ...
                                              cross(T(1:3,2), T_final(1:3,2)) + ...
                                              cross(T(1:3,3), T_final(1:3,3)));
                                    e = [ex; eo];   
                                elseif strcmp(jacobian_type, "numerical")    
                                    e = D_final - D_current; 
                                end
                                
                            end
                            
                            %norm(e)                                                       
                            %% track the number of iterations
                            current_iteration = current_iteration + 1;
                            
                            if strcmp(mode_print, "True")
                                fprintf("\n\nCurrent iteration [%d]\n", current_iteration);  
                            end
                            
                            %% stopping criteria to break out of the while: maximum_iteration or desired error reached
                            %{
                            if or(norm(e)< emax, current_iteration > maximum_iteration)
                                final = 1;
                            end
                            %}
                            
                            if strcmp(robot, 'RRP')
                                if (abs(D_final(1) - D_current(1)) < position_error) && (abs(D_final(2) - D_current(2)) < position_error)  || current_iteration > maximum_iteration 
                                    final = 1;
                                end
                            end
                            
                            if strcmp(robot, 'RRPRRRR') || strcmp(robot, 'RRRRRRR') || strcmp(robot, 'RRVPRVRVPRR')
                                if (abs(D_final(1) - D_current(1)) < position_error) && (abs(D_final(2) - D_current(2)) < position_error) && (abs(D_final(3) - D_current(3)) < position_error) && (abs(D_final(4) - D_current(4)) < orientation_error) && (abs(D_final(5) - D_current(5)) < orientation_error) && (abs(D_final(6) - D_current(6)) < orientation_error) || current_iteration > maximum_iteration 
                                    final = 1;
                                end
                            end
                            
                            
                            %% save path
                            if strcmp(mode_save_path, "True")
                                saveToFile(D_current, motion, str_unit_chosen, inverse_chosen, alpha, "a");
                            end
        
                            
                            
        
        
                        end % end while loop for maximum iterations
        
                        %% check if the solution was found before recording the results
                        if (current_iteration < maximum_iteration)
                            solutionFound = 1;
                        else
                            solutionFound = 0;
                        end 
        
                        if strcmp(mode_print_result, "True")
                            %% summary of the IK search solution
                            fprintf("######################################################\n\n")
                            fprintf("Estimated Joint Vector:")
                            if strcmp(robot, 'RRVPRVRVPRR')
                                Q_current(3) = 0;
                                Q_current(6) = 0;
                                Q_current(8) = 0;
                                Q_current %#ok<NOPTS
                            else
                                Q_current %#ok<NOPTS
                            end
        
                            fprintf("Initial Joint Vector:")
                            Q_initial %#ok<NOPTS
        
                            fprintf("Initial Pose Vector:")
                            D_initial %#ok<NOPTS
        
                            fprintf("Desired Pose Vector:")
                            D_final %#ok<NOPTS>
        
                            fprintf("Estimated Pose Vector:")
                            D_current %#ok<NOPTS>
                        end
                        
                            if strcmp(robot, "RRP")
                                Position_error = sqrt(sum((D_final - D_current).^2));
                                %fprintf("Position error: %f mm", (Position_error*unit_applied));
                            else
                                Position_error = sqrt(sum((D_final(1:3) - D_current(1:3)).^2));
                                fprintf("Position error: %f mm", (Position_error*unit_applied));
        
                                Orientation_error = sum(abs(D_final(4:6) - D_current(4:6)))/3;
                                fprintf("\nOrientation error: %f degrees", rad2deg(Orientation_error));
                            end
        
                            fprintf("\n\nTotal number of iterations: %d", current_iteration); 
                            fprintf("\n==> Done with Sample [%d]\n", s);
                        
                         
                        %fprintf("\n")
                        %toc %finishes recording time
                        elapsedTime = toc;
                        
                        
                        
                        %% save the summary results in a table
                        if strcmp(robot, 'RRP')
                            ideal_distance = sqrt(sum((D_final - D_initial).^2));
                            summaryTable(s, :) = [Q_initial; ...
                                                    D_initial; ...
                                                    D_final; ...
                                                    D_current; ...
                                                    Position_error; ...
                                                    current_iteration; elapsedTime; solutionFound; ...
                                                    ideal_distance; performed_distance];  
                        end
        
                        if strcmp(robot, 'RRRRRRR')
                            ideal_distance = sqrt(sum((D_final(1:3) - D_initial(1:3)).^2));
                            summaryTable(s, :) = [Q_initial; ...
                                                    D_initial; ...
                                                    D_final; ...
                                                    D_current; ...
                                                    Position_error; Orientation_error; ...
                                                    current_iteration; elapsedTime; solutionFound; ...
                                                    ideal_distance; performed_distance; ss];  
                        end
        
                        if strcmp(robot, 'RRPRRRR') || strcmp(robot, 'RRVPRVRVPRR')
                            ideal_distance = sqrt(sum((D_final(1:3) - D_initial(1:3)).^2));
                            summaryTable(s, :) = [Q_initial; ...
                                                    D_initial; ...
                                                    D_final; ...
                                                    D_current; ...
                                                    Position_error; Orientation_error; ...
                                                    current_iteration; elapsedTime; solutionFound; ...
                                                    ideal_distance; performed_distance; ss];  
                        end
                        
                        
                        total_computation = total_computation + elapsedTime;
                        
                    end % end of the samples loop
                                          
                            
                
                
                  
                  
                    %% save the tables of the results to be analyzed
                    if strcmp(mode_save, "True")
                        if strcmp(robot, 'RRP')
                            varNames = {'Q1', 'Q2', 'Q3', ...
                                        'D1_initial', 'D2_initial', ...
                                        'D1_final', 'D2_final', ...
                                        'D1_estimated', 'D2_estimated', ...
                                        'Final_position_error', 'Total_iterations', ...
                                        'Computation_time', 'Solution_found', ...
                                        'ideal_distance', 'performed_distance'};
        
                            dataset_t = table(summaryTable(:,1), summaryTable(:,2), summaryTable(:,3), ...
                                              summaryTable(:,4), summaryTable(:,5), ...
                                              summaryTable(:,6), summaryTable(:,7), ...
                                              summaryTable(:,8), summaryTable(:,9),...
                                              summaryTable(:,10), summaryTable(:,11),...
                                              summaryTable(:,12), summaryTable(:,13),...
                                              summaryTable(:,14), summaryTable(:,15), ...
                                              'VariableNames',varNames);
                            writetable(dataset_t, strcat(motion,'/','results_',robot_call,'_',str_unit_chosen,'_', inverse_chosen ,'_', num2str(alpha) ,'_seq_',num2str(seq),'.csv'),'Delimiter',',')
                        end
        
                        if strcmp(robot, 'RRRRRRR')
                            varNames = {'Q1', 'Q2', 'Q3','Q4', 'Q5', 'Q6','Q7', ...
                                        'D1_initial', 'D2_initial','D3_initial', 'D4_initial','D5_initial', 'D6_initial', ...
                                        'D1_final', 'D2_final','D3_final', 'D4_final','D5_final', 'D6_final', ...
                                        'D1_estimated', 'D2_estimated','D3_estimated', 'D4_estimated','D5_estimated', 'D6_estimated', ...
                                        'Final_position_error', 'Final_orientation_error','Total_iterations', ...
                                        'Computation_time', 'Solution_found', ...
                                        'ideal_distance', 'performed_distance', 'singularity'};
        
                            dataset_t = table(summaryTable(:,1), summaryTable(:,2), summaryTable(:,3), summaryTable(:,4), summaryTable(:,5), summaryTable(:,6), summaryTable(:,7), ...
                                              summaryTable(:,8), summaryTable(:,9), summaryTable(:,10), summaryTable(:,11), summaryTable(:,12), summaryTable(:,13),...
                                              summaryTable(:,14), summaryTable(:,15), summaryTable(:,16), summaryTable(:,17), summaryTable(:,18), summaryTable(:,19), ...
                                              summaryTable(:,20), summaryTable(:,21), summaryTable(:,22), summaryTable(:,23), summaryTable(:,24), summaryTable(:,25),...
                                              summaryTable(:,26), summaryTable(:,27),...
                                              summaryTable(:,28), summaryTable(:,29),...
                                              summaryTable(:,30), summaryTable(:,31),summaryTable(:,32), summaryTable(:,33),...
                                              'VariableNames',varNames);
                            writetable(dataset_t, strcat(motion,'/','results_',robot_call,'_',str_unit_chosen,'_', inverse_chosen ,'_', num2str(alpha) ,'_seq_',num2str(seq),'.csv'),'Delimiter',',')
                        end
        
                        if strcmp(robot, 'RRPRRRR')
                            varNames = {'Q1', 'Q2', 'Q3','Q4', 'Q5', 'Q6','Q7', ...
                                        'D1_initial', 'D2_initial','D3_initial', 'D4_initial','D5_initial', 'D6_initial', ...
                                        'D1_final', 'D2_final','D3_final', 'D4_final','D5_final', 'D6_final', ...
                                        'D1_estimated', 'D2_estimated','D3_estimated', 'D4_estimated','D5_estimated', 'D6_estimated', ...
                                        'Final_position_error', 'Final_orientation_error','Total_iterations', ...
                                        'Computation_time', 'Solution_found', ...
                                        'ideal_distance', 'performed_distance', 'singularity'};
        
                            dataset_t = table(summaryTable(:,1), summaryTable(:,2), summaryTable(:,3), summaryTable(:,4), summaryTable(:,5), summaryTable(:,6), summaryTable(:,7), ...
                                              summaryTable(:,8), summaryTable(:,9), summaryTable(:,10), summaryTable(:,11), summaryTable(:,12), summaryTable(:,13),...
                                              summaryTable(:,14), summaryTable(:,15), summaryTable(:,16), summaryTable(:,17), summaryTable(:,18), summaryTable(:,19), ...
                                              summaryTable(:,20), summaryTable(:,21), summaryTable(:,22), summaryTable(:,23), summaryTable(:,24), summaryTable(:,25),...
                                              summaryTable(:,26), summaryTable(:,27),...
                                              summaryTable(:,28), summaryTable(:,29),...
                                              summaryTable(:,30), summaryTable(:,31),summaryTable(:,32),summaryTable(:,33), ...
                                              'VariableNames',varNames);
                            writetable(dataset_t, strcat(motion,'/','results_',robot_call,'_',str_unit_chosen,'_', inverse_chosen ,'_', num2str(alpha), '_seq_',num2str(seq),'.csv'),'Delimiter',',')
                        end
                        
                        if strcmp(robot, 'RRVPRVRVPRR')
                            varNames = {'Q1', 'Q2', 'Q3','Q4', 'Q5', 'Q6','Q7','Q8','Q9','Q10','Q11' ...
                                        'D1_initial', 'D2_initial','D3_initial', 'D4_initial','D5_initial', 'D6_initial', ...
                                        'D1_final', 'D2_final','D3_final', 'D4_final','D5_final', 'D6_final', ...
                                        'D1_estimated', 'D2_estimated','D3_estimated', 'D4_estimated','D5_estimated', 'D6_estimated', ...
                                        'Final_position_error', 'Final_orientation_error','Total_iterations', ...
                                        'Computation_time', 'Solution_found', ...
                                        'ideal_distance', 'performed_distance', 'singularity'};
        
                            dataset_t = table(summaryTable(:,1), summaryTable(:,2), summaryTable(:,3), summaryTable(:,4), summaryTable(:,5), summaryTable(:,6), summaryTable(:,7), ...
                                              summaryTable(:,8), summaryTable(:,9), summaryTable(:,10), summaryTable(:,11), summaryTable(:,12), summaryTable(:,13), ...
                                              summaryTable(:,14), summaryTable(:,15), summaryTable(:,16), summaryTable(:,17), summaryTable(:,18), summaryTable(:,19), ...
                                              summaryTable(:,20), summaryTable(:,21), summaryTable(:,22), summaryTable(:,23), summaryTable(:,24), summaryTable(:,25), ...
                                              summaryTable(:,26), summaryTable(:,27), summaryTable(:,28), summaryTable(:,29), summaryTable(:,30), summaryTable(:,31), ...,
                                              summaryTable(:,32),summaryTable(:,33), summaryTable(:,34), summaryTable(:,35), summaryTable(:,36), summaryTable(:,37), ...
                                              'VariableNames',varNames);
                            writetable(dataset_t, strcat(motion,'/','results_',robot_call,'_',str_unit_chosen,'_', inverse_chosen ,'_', num2str(alpha) ,'_seq_',num2str(seq),'.csv'),'Delimiter',',')
                        end
                    end
                end % end of the alpha loop 
                
            end   % end of the units loop
        end % end of the inverse loop
        

    end

    fprintf("\n\nTotal computation time (minutes):")
    total_computation = total_computation/60 %#ok<NOPTS>

end