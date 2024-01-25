%% load data
%data = readmatrix("../docker/datasets/7DoF-7R-Panda/data_7DoF-7R-Panda_1000000_qlim_scale_10_seq.csv");

total = 20;
for i=1:total
    data = readmatrix(strcat("../docker/datasets/7DoF-7R-Panda-Steps/data_7DoF-7R-Panda_1000000_qlim_scale_10_seq_",num2str(i),".csv"));

    %% get mean RPY
    mean_xyzRPY = mean(abs(data(:,1:6) - data(:,14:19)), 1);
    mean_xyzRPY(1:3) = mean_xyzRPY(1:3)*1000;
    mean_xyzRPY(4:6) = rad2deg(mean_xyzRPY(4:6));

    fprintf("Joint variation: %f", i)
    mean_xyzRPY  %#ok<NOPTS>
end