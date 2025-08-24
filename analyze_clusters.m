%% Cluster Analysis Script
% This script loads your saved results and shows cluster statistics
% Run this after running script.m or batch_process_folders.m

clear; clc;

%% Load Results
% Change this path to match your output file location
results_file = 'output/final_results.mat';  % Single file results
% results_file = 'batch_output/folder_name/filename_final_results.mat';  % Batch results

if ~exist(results_file, 'file')
    fprintf('Results file not found: %s\n', results_file);
    fprintf('Please run the main script first, or check the file path.\n');
    return;
end

fprintf('Loading results from: %s\n', results_file);
data = load(results_file);

%% Display Basic Information
fprintf('\n=== BASIC INFORMATION ===\n');
fprintf('Total calls detected: %d\n', height(data.Calls));
fprintf('Valid spectrograms: %d\n', height(data.ClusteringData));

%% Check if clustering was performed
if isfield(data.ClusteringData, 'Cluster')
    fprintf('\n=== CLUSTER ANALYSIS ===\n');
    
    % Get cluster assignments
    cluster_assignments = data.ClusteringData.Cluster;
    
    % Remove failed calls (marked as -1)
    valid_clusters = cluster_assignments(cluster_assignments > 0);
    
    if ~isempty(valid_clusters)
        % Get unique cluster IDs
        unique_clusters = unique(valid_clusters);
        num_clusters = length(unique_clusters);
        
        fprintf('Number of clusters: %d\n', num_clusters);
        fprintf('Successfully clustered calls: %d\n', length(valid_clusters));
        
        % Display cluster statistics
        fprintf('\n--- CLUSTER BREAKDOWN ---\n');
        for i = 1:num_clusters
            cluster_id = unique_clusters(i);
            count = sum(cluster_assignments == cluster_id);
            percentage = 100 * count / length(valid_clusters);
            
            fprintf('Cluster %d: %d calls (%.1f%%)\n', cluster_id, count, percentage);
        end
        
        % Show failed calls
        failed_calls = sum(cluster_assignments == -1);
        if failed_calls > 0
            fprintf('\nFailed calls: %d (%.1f%%)\n', failed_calls, 100*failed_calls/height(data.ClusteringData));
        end
        
        %% Detailed Cluster Information
        fprintf('\n--- DETAILED CLUSTER INFO ---\n');
        for i = 1:num_clusters
            cluster_id = unique_clusters(i);
            cluster_indices = find(cluster_assignments == cluster_id);
            
            fprintf('\nCluster %d (%d calls):\n', cluster_id, length(cluster_indices));
            
            % Show first few call details
            for j = 1:min(3, length(cluster_indices))
                call_idx = cluster_indices(j);
                call_box = data.ClusteringData.Box(call_idx, :);
                fprintf('  Call %d: Time=%.2fs, Freq=%.1fkHz, Duration=%.3fs, Bandwidth=%.1fkHz\n', ...
                    call_idx, call_box(1), call_box(2), call_box(3), call_box(4));
            end
            
            if length(cluster_indices) > 3
                fprintf('  ... and %d more calls\n', length(cluster_indices) - 3);
            end
        end
        
        %% Cluster Distances (if available)
        if isfield(data.ClusteringData, 'ClusterDistance')
            fprintf('\n--- CLUSTER DISTANCES ---\n');
            cluster_distances = data.ClusteringData.ClusterDistance;
            
            for i = 1:num_clusters
                cluster_id = unique_clusters(i);
                cluster_indices = find(cluster_assignments == cluster_id);
                distances = cluster_distances(cluster_indices);
                
                fprintf('Cluster %d: Mean distance = %.3f, Std = %.3f\n', ...
                    cluster_id, mean(distances), std(distances));
            end
        end
        
        %% Export Results to CSV (optional)
        export_csv = input('\nExport cluster results to CSV? (y/n): ', 's');
        if strcmpi(export_csv, 'y') || strcmpi(export_csv, 'yes')
            % Create export table
            export_data = table();
            export_data.CallIndex = (1:height(data.ClusteringData))';
            export_data.Cluster = cluster_assignments;
            export_data.Time = data.ClusteringData.Box(:, 1);
            export_data.Frequency = data.ClusteringData.Box(:, 2);
            export_data.Duration = data.ClusteringData.Box(:, 3);
            export_data.Bandwidth = data.ClusteringData.Box(:, 4);
            
            if isfield(data.ClusteringData, 'ClusterDistance')
                export_data.Distance = data.ClusteringData.ClusterDistance;
            end
            
            % Save CSV
            csv_filename = 'cluster_results.csv';
            writetable(export_data, csv_filename);
            fprintf('Results exported to: %s\n', csv_filename);
        end
        
    else
        fprintf('No valid clusters found in the data.\n');
    end
    
else
    fprintf('\nNo clustering data found. The script may not have completed clustering.\n');
    fprintf('Check if your VAE network loaded successfully.\n');
end

%% Show available data fields
fprintf('\n=== AVAILABLE DATA FIELDS ===\n');
fprintf('Calls table fields: %s\n', strjoin(data.Calls.Properties.VariableNames, ', '));
fprintf('ClusteringData table fields: %s\n', strjoin(data.ClusteringData.Properties.VariableNames, ', '));

%% Quick visualization (if you have the Image Processing Toolbox)
try
    if isfield(data.ClusteringData, 'Cluster') && exist('histogram', 'file')
        figure('Name', 'Cluster Distribution');
        
        % Create histogram of cluster assignments
        valid_clusters = cluster_assignments(cluster_assignments > 0);
        histogram(valid_clusters, 'FaceColor', 'skyblue', 'EdgeColor', 'black');
        xlabel('Cluster ID');
        ylabel('Number of Calls');
        title('Distribution of Calls Across Clusters');
        grid on;
        
        fprintf('\nCluster distribution histogram displayed.\n');
    end
catch ME
    fprintf('\nCould not create visualization: %s\n', ME.message);
end

fprintf('\n=== ANALYSIS COMPLETE ===\n');
