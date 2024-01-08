data = readmatrix("../docker/datasets/7DoF-7R-Panda/data_7DoF-7R-Panda_100000000_qlim_scale_10.csv");
                      %docker/datasets/7DoF-7R-Panda/data_7DoF-7R-Panda_1000000_qlim_scale_10.csv

threshold_position = 0.1; %0.1;
threshold_orientation = deg2rad(5);
[total_samples, ~] = size(data);

% just consider a subset of the total samples for debugging (comment to run everything)
%total_samples = 10000;
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
    %distances = sqrt(sum((pose(1:total_samples,1:3)-pose(i,1:3)).^2, 2));

    % Build the H of the current traversed sample
    cH = eye(4);
    cH(1:3,1:3) = eul2rotm(pose(i,4:6), "XYZ");
    cH(1:3,4) = pose(i,1:3);
        
    % Build the Hs of all the data points
    allH = repmat(eye(4), 1, 1, total_samples);
    allH(1:3,1:3,:) = eul2rotm(pose(:,4:6), "XYZ");
    allH(1:3,4,:) = pose(:,1:3)';
    
    % Find the inverse H of all the remaining points in the dataset 
    %{
    invallH =  repmat(eye(4), 1, 1, total_samples);
    invallH(1:3,1:3,:) = pagetranspose(allH(1:3,1:3,:));
    invallH(1:3,4,:) = pagemtimes(-pagetranspose(allH(1:3,1:3,:)),allH(1:3,4,:));
    %}
    
    invcH = eye(4);
    invcH(1:3,1:3,:) = cH(1:3,1:3,:)';
    invcH(1:3,4,:) = (-cH(1:3,1:3)')*cH(1:3,4);
    
    % Find the difference in H form between the current data point and all
    % the data points in the dataset
    %distH = pagemtimes(cH,invallH);    
    distH = pagemtimes(allH, invcH);
    
    % Remove the distance of the current point to itself
    distH(:,:,i) = []; 
    distPosition = sqrt(sum(distH(1:3,4,:).^2, 1));
    distPosition = squeeze(distPosition);
          
    %distH2(:,:,i) = []; 
    %distPosition2 = sqrt(sum(distH2(1:3,4,:).^2, 1));
    %distPosition2 = squeeze(distPosition2);
    
    % Count all the distances less than a distance threshold
    idx = find(distPosition<threshold_position);
    %idx = idx + 1;
    
    %idx = find(distPosition2<threshold_position);
    %idx2 = idx2 + 1;
    
    % Now that you have found the points satisfying the position threshod 
    % find the points that satisfy the orientation threshold
    distOrientation = abs(rotm2eul(distH(1:3,1:3,:), "XYZ"));
    IdxOr = distOrientation(idx,:) < threshold_orientation;
    IdxOr = sum(IdxOr, 2);
    IdxOr = find(IdxOr == 3);                                  % check that the 3 RPY error values are less than the orientation threshold
    IdxRemove = idx(IdxOr);  
    IdxRemove = IdxRemove+1;

    % Remove the points that satifies the position and orientation
    % threshold
    pose(IdxRemove,:) = [];
    data_save(IdxRemove,:) = [];
    
    [total_samples,~] = size(data_save);
    
    count = count + numel(IdxRemove);
    
    fprintf('\n')
    disp(['number of indices removed in current [i]: ', num2str(numel(IdxRemove))])
    disp(['total samples already removed so far: ', num2str(count)])
    disp(['current traversed index among the remaining samples [i]: ', num2str(i)])
    disp(['total remaining samples: ', num2str(total_samples)])
    
    %if mod(i,1000) == 0
    %    fprintf('\n')
    %    toc
    %    disp(['Current sample: ', num2str(i)]);       
    %end
    
    if i >= total_samples
        disp("Stopping")
        break
    end 
    
    if i == 1000000
        disp("Stopping ... Reached 1 million points")
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

%tic
%s_distances = pdist(data_save(: ,1:3), 'euclidean');
%s_count = sum(s_distances < threshold_position)
%toc


%% plot points
f = figure(2);
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


%varNames = {'x','y','z','R','P','Y','t1','t2','t3','t4','t5','t6','t7'};
%dataToSave = 


writematrix(data_save,'data_save_7DoF_from_samples_1000000_true.csv','Delimiter',',')


