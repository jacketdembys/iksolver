data = readmatrix("../docker/datasets/7DoF-7R-Panda/data_7DoF-7R-Panda_1000000_qlim_scale_10.csv");
                      %docker/datasets/7DoF-7R-Panda/data_7DoF-7R-Panda_1000000_qlim_scale_10.csv

threshold_position = 0.1;
threshold_orientation = 5;
[total_samples, ~] = size(data);

% just consider a subset of the total samples for debugging (comment to run everything)
total_samples = 1000;
total_samples_save = total_samples;
data_save = data(1:total_samples,:);
pose = data(1:total_samples,1:6);
pose(:,1:3) = pose(:,1:3);
pose(:,4:6) = pose(:,4:6);



%% test with custom script 
% initialize counter
count = 0;

tallstart = tic;
tic
% loop through the samples
%for i=1:total_samples
i = 1;

while(1)
    
    % compute the pairwise distances
    distances = sqrt(sum((pose(1:total_samples,1:3)-pose(i,1:3)).^2, 2));

    % remove the pair with two identical values of the current element
    distances(i) = [];

    % count all the distances less than a distance threshold
    %count = count + sum(distances<threshold_position);
    idx = find(distances<threshold_position);
    idx = idx + 1;
    %numel(distances(idx))
    %count = count + numel(distances(idx));
    pose(idx,:) = [];
    data_save(idx,:) = [];
    
    [total_samples,~] = size(data_save);
    
    fprintf('\n')
    disp(['number of indices: ', num2str(numel(idx))])
    disp(['i: ', num2str(i)])
    disp(['total_samples: ', num2str(total_samples)])
    
    %if mod(i,1000) == 0
    %    fprintf('\n')
    %    toc
    %    disp(['Current sample: ', num2str(i)]);       
    %end
    
    if i >= total_samples
        disp("Stopping")
        break
    end    
    
    i = i + 1;
    
end
elapsed_time = toc(tallstart);
disp(['Elapsed time: ', num2str(elapsed_time), ' seconds'])

%total_distances = total_samples;
%total_distances = ((total_samples^2)-total_samples)/2;
%count_report = ((count/2)/total_distances)*100


%% report results 
count_report = ((total_samples_save-total_samples)/total_samples_save)*100

tic
s_distances = pdist(data_save(: ,1:3), 'euclidean');
s_count = sum(s_distances < threshold_position)
toc


%% plot points
f = figure(1);
f.Position = [300 300 1200 500];
tiledlayout(1,2)

nexttile
scatter3(data(:,1),data(:,2),data(:,3), '.')
xlabel('X(m)')
ylabel('Y(m)')
zlabel('Z(m)')
set(gca, 'FontSize', 18)
title({'Original Dataset', strcat('(',num2str(total_samples_save),' samples)')})

nexttile
scatter3(data_save(:,1),data_save(:,2),data_save(:,3), '.')
xlabel('X(m)')
ylabel('Y(m)')
zlabel('Z(m)')
set(gca, 'FontSize', 16)
title({strcat("Constrained Dataset with threshold = ", num2str(threshold_position*1000), " mm"), strcat('(',num2str(total_samples),' samples remaining \approx ', " ", num2str(round(count_report,2)),'% removed)')})


%% save workspace for later analysis
%filename = "remove_count_less_than_100.mat";
%save(filename)


