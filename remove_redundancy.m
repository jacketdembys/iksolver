data = readmatrix("../docker/datasets/7DoF-7R-Panda/data_7DoF-7R-Panda_1000000_qlim_scale_10.csv");


threshold_position = 100;
threshold_orientation = 5;
[total_samples, ~] = size(data);

pose = data(:,1:6);
pose(:,1:3) = pose(:,1:3)*1000;
pose(:,4:6) = rad2deg(pose(:,4:6));


tic 
% initialize counter
count = 0;

% just consider a subset
total_samples = 1000;

% loop through the samples
for i=1:total_samples

    % compute the pairwise distances
    distances = sqrt(sum((pose(1:total_samples,1:3)-pose(i,1:3)).^2, 2));

    % remove the pair with two identical values of the current element
    distances(i) = [];

    % count all the distances less than a distance threshold
    %count = count + sum(distances<threshold_position);
    idx = find(distances<threshold_position);
    count = count + numel(distances(idx));
end
toc

count
count_report = ((count/2)/total_samples)*100



%%% Test with 
distances = pdist(pose(1:total_samples ,1:3), 'euclidean');
count = sum(distances < threshold_position) 
count_report_2 = (count/total_samples)*100



















