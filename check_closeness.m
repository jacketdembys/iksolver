%% load data
%data = readmatrix("../docker/datasets/7DoF-7R-Panda/data_7DoF-7R-Panda_1000000_qlim_scale_10_seq.csv");

total = 21; %20
mean_xyzRPY_save = zeros(total,6);
robot_choice = "7DoF-7R-Panda"; % 7DoF-7R-Panda
for i=1:total
    
    data = readmatrix(strcat("../docker/datasets/",robot_choice,"-Steps/data_",robot_choice,"_1000000_qlim_scale_10_seq_",num2str(i-1),".csv"));

    %% get mean RPY
    mean_xyzRPY = mean(abs(data(:,1:6) - data(:,14:19)), 1);
    mean_xyzRPY(1:3) = mean_xyzRPY(1:3)*1000;
    mean_xyzRPY(4:6) = rad2deg(mean_xyzRPY(4:6));

    fprintf("Joint variation: %f", i-1)
    mean_xyzRPY  %#ok<NOPTS>
    mean_xyzRPY_save(i,:) = mean_xyzRPY;
    
end

filename = strcat("mean_xyzRPY_", robot_choice, ".csv");
writematrix(mean_xyzRPY_save,filename)