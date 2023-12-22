data = readmatrix("../docker/datasets/7DoF-7R-Panda/data_7DoF-7R-Panda_1000000_qlim_scale_10.csv");

threshold_position = 100;
chunkSize = 1000; % Define the chunk size for processing

[total_samples, ~] = size(data);
pose = data(:, 1:6);
pose(:, 1:3) = pose(:, 1:3) * 1000;

tic
count_position = 0;

total_samples = 1000;

for i = 1:chunkSize:total_samples
    endIndex = min(i + chunkSize - 1, total_samples);
    current_chunk = pose(i:endIndex, :);

    % Calculate pairwise distances for the current chunk
    distances = pdist(current_chunk(:, 1:3), 'euclidean');
    
    % Count distances below the threshold
    count_position = count_position + sum(distances < threshold_position);
end

toc

count_report = (count_position/total_samples)*100;

disp(['Number of positions below the threshold: ', num2str(count_position)]);
disp(['Percentage of redundant positions within the threshold: ', num2str(count_report)]);
