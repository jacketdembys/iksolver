data = readmatrix("../docker/datasets/7DoF-7R-Panda/data_7DoF-7R-Panda_1000000_qlim_scale_10.csv");


threshold_position = 100;
threshold_orientation = 5;
[total_samples, ~] = size(data);

% just consider a subset of the total samples for debugging (comment to run everything)
%total_samples = 10000;

pose = data(:,1:6);
pose(:,1:3) = pose(:,1:3)*1000;
pose(:,4:6) = rad2deg(pose(:,4:6));



%% test with custom script 
% initialize counter
count = 0;

tallstart = tic;
tic
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
    
    if mod(i,1000) == 0
        fprintf('\n')
        toc
        disp(['Current sample: ', num2str(i)]);       
    end
    
    
end
elapsed_time = toc(tallstart);
disp(['Elapsed time: ', num2str(elapsed_time), ' seconds'])

%total_distances = total_samples;
total_distances = ((total_samples^2)-total_samples)/2;
count_report = ((count/2)/total_distances)*100


%{
%% Test with the in-build "pdist" function (might get stuck building a 1e6*1e6)
tic
distances = pdist(pose(1:total_samples ,1:3), 'euclidean');
count = sum(distances < threshold_position); 
toc

total_distances = numel(distances);
count_report = (count/total_distances)*100



%% Test with the in-build "pdist" function but specified in chunks of array
chunkSize = 1000; % Define the chunk size for processing

count_position = 0;
tic
for i = 1:chunkSize:total_samples
    % keep track of the different chuncks of the array
    endIndex = min(i + chunkSize - 1, total_samples);
    current_chunk = pose(i:endIndex, :);

    % Calculate pairwise distances for the current chunk
    distances = pdist(current_chunk(:, 1:3), 'euclidean');
    
    % Count distances below the threshold
    count_position = count_position + sum(distances < threshold_position);
    %idx = find(distances<threshold_position);
    %count_position = count_position + numel(distances(idx));
end
toc

total_distances = ((total_samples^2)-total_samples)/2;
count_report = (count_position/total_distances)*100
%}



















