%% DeepSqueak Batch Processing Script
% This script processes multiple audio folders sequentially
% It runs the main processing script on each folder one at a time

clear; clc; close all;

%% Configuration
% Load configuration from external file
if exist('batch_config.m', 'file')
    run('batch_config.m');
    fprintf('Loaded configuration from batch_config.m\n');
else
    fprintf('Warning: batch_config.m not found. Using default settings.\n');
    
    % Default configuration
    base_directory = 'Audio/';
    output_base_directory = 'batch_output/';
    detection_settings = {'0', '100', '18', '0.5', '1'};
    network_file = 'Networks/2025.02.18.Mouse.YoloR3.mat';
    clustering_method = 'VAE';
    save_results = true;
    custom_network_path = 'NNTest#1.mat';
    use_parallel = false;
    max_files_per_folder = 0;
    skip_existing_output = false;
    create_detailed_logs = true;
    max_audio_duration = 300;
end

%% Find all audio folders
fprintf('Scanning for audio folders in: %s\n', base_directory);

% Get all subdirectories
all_dirs = dir(base_directory);
audio_folders = {};

for i = 1:length(all_dirs)
    if all_dirs(i).isdir && ~strcmp(all_dirs(i).name, '.') && ~strcmp(all_dirs(i).name, '..')
        folder_path = fullfile(base_directory, all_dirs(i).name);
        
        % Check if folder contains audio files
        audio_files = dir(fullfile(folder_path, '*.wav'));
        audio_files = [audio_files; dir(fullfile(folder_path, '*.flac'))];
        audio_files = [audio_files; dir(fullfile(folder_path, '*.mp3'))];
        audio_files = [audio_files; dir(fullfile(folder_path, '*.m4a'))];
        
        if ~isempty(audio_files)
            audio_folders{end+1} = all_dirs(i).name;
            fprintf('Found audio folder: %s (%d audio files)\n', all_dirs(i).name, length(audio_files));
        end
    end
end

% Also check the base directory for audio files
base_audio_files = dir(fullfile(base_directory, '*.wav'));
base_audio_files = [base_audio_files; dir(fullfile(base_directory, '*.flac'))];
base_audio_files = [base_audio_files; dir(fullfile(base_directory, '*.mp3'))];
base_audio_files = [base_audio_files; dir(fullfile(base_directory, '*.m4a'))];

if ~isempty(base_audio_files)
    audio_folders{end+1} = '.';  % Base directory
    fprintf('Found audio files in base directory: %d files\n', length(base_audio_files));
end

if isempty(audio_folders)
    fprintf('No audio folders found. Exiting.\n');
    return;
end

fprintf('\nTotal audio folders to process: %d\n', length(audio_folders));

%% Process each folder sequentially
for folder_idx = 1:length(audio_folders)
    current_folder = audio_folders{folder_idx};
    
    fprintf('\n========================================\n');
    fprintf('Processing folder %d/%d: %s\n', folder_idx, length(audio_folders), current_folder);
    fprintf('========================================\n');
    
    % Create output directory for this folder
    if strcmp(current_folder, '.')
        output_folder = fullfile(output_base_directory, 'base_directory');
    else
        output_folder = fullfile(output_base_directory, current_folder);
    end
    
    if ~exist(output_folder, 'dir')
        mkdir(output_folder);
    end
    
    % Get all audio files in current folder
    if strcmp(current_folder, '.')
        folder_path = base_directory;
    else
        folder_path = fullfile(base_directory, current_folder);
    end
    
    audio_files = dir(fullfile(folder_path, '*.wav'));
    audio_files = [audio_files; dir(fullfile(folder_path, '*.flac'))];
    audio_files = [audio_files; dir(fullfile(folder_path, '*.mp3'))];
    audio_files = [audio_files; dir(fullfile(folder_path, '*.m4a'))];
    
    fprintf('Found %d audio files in folder\n', length(audio_files));
    
    % Process each audio file in the current folder
    for file_idx = 1:length(audio_files)
        audio_file = fullfile(folder_path, audio_files(file_idx).name);
        
        fprintf('\n--- Processing file %d/%d: %s ---\n', file_idx, length(audio_files), audio_files(file_idx).name);
        
        try
            % Process this audio file using the main script logic
            process_single_audio_file(audio_file, network_file, output_folder, detection_settings, clustering_method, save_results, custom_network_path);
            
            fprintf('Successfully processed: %s\n', audio_files(file_idx).name);
            
        catch ME
            fprintf('ERROR processing %s: %s\n', audio_files(file_idx).name, ME.message);
            fprintf('Continuing with next file...\n');
            
            % Save error information
            error_file = fullfile(output_folder, sprintf('error_%s.txt', audio_files(file_idx).name));
            fid = fopen(error_file, 'w');
            fprintf(fid, 'Error processing file: %s\n', audio_files(file_idx).name);
            fprintf(fid, 'Error message: %s\n', ME.message);
            fprintf(fid, 'Error location: %s\n', ME.stack(1).name);
            fclose(fid);
        end
    end
    
    fprintf('\nCompleted processing folder: %s\n', current_folder);
    fprintf('Output saved to: %s\n', output_folder);
end

%% Summary
fprintf('\n========================================\n');
fprintf('BATCH PROCESSING COMPLETE\n');
fprintf('========================================\n');
fprintf('Processed %d folders\n', length(audio_folders));
fprintf('Output base directory: %s\n', output_base_directory);
fprintf('\nBatch processing completed successfully!\n');

%% Function to process a single audio file
function process_single_audio_file(audio_file, network_file, output_folder, detection_settings, clustering_method, save_results, custom_network_path)
%% Process a single audio file using DeepSqueak functions
% This function contains the core processing logic from the main script

%% Initialize
% Initialize squeakData object (minimal setup for headless operation)
handles = struct();
handles.data = struct();
handles.data.settings = struct();
handles.data.settings.detectionfolder = pwd;
handles.data.settings.networkfolder = fullfile(pwd, 'Networks/');
handles.data.settings.audiofolder = fullfile(pwd, 'Audio/');
handles.data.settings.EntropyThreshold = 0.5;
handles.data.settings.AmplitudeThreshold = 0.1;

% Load audio file info
if ~exist(audio_file, 'file')
    error('Audio file not found: %s', audio_file);
end

audio_info = audioinfo(audio_file);
fprintf('Audio file: %s\n', audio_file);
fprintf('Duration: %.2f seconds\n', audio_info.Duration);
fprintf('Sample rate: %d Hz\n', audio_info.SampleRate);
fprintf('Channels: %d\n', audio_info.NumChannels);

%% Load Neural Network
fprintf('Loading neural network...\n');
if ~exist(network_file, 'file')
    error('Network file not found: %s', network_file);
end

network_data = load(network_file);
fprintf('Loaded network: %s\n', network_file);

%% Detect Calls
fprintf('Running call detection...\n');

% Convert settings to numeric values
settings = zeros(1, length(detection_settings));
for i = 1:length(detection_settings)
    settings(i) = str2double(detection_settings{i});
end

% Detect calls using SqueakDetect function
tic;
Calls = SqueakDetect(audio_file, network_data, 'Audio File', settings, 1, 1, 'Detection Network');
detection_time = toc;

fprintf('Detection completed in %.2f seconds\n', detection_time);
fprintf('Found %d calls\n', height(Calls));

if isempty(Calls)
    fprintf('No calls detected. Skipping this file.\n');
    return;
end

%% Create Spectrograms
fprintf('Creating call spectrograms...\n');

% Initialize audio reader
audioReader = squeakData([]);
audioReader.audiodata = audio_info;

% Create spectrograms for each detected call
call_spectrograms = cell(height(Calls), 1);
call_boxes = Calls.Box;

for i = 1:height(Calls)
    % Create call object for spectrogram generation
    call = struct();
    call.Box = call_boxes(i, :);
    call.RelBox = call_boxes(i, :);
    call.Rate = audio_info.SampleRate;
    
    % Create focus spectrogram around the call
    try
        [I, ~, ~, ~, ~, ~, ~, ~, ~, ~] = CreateFocusSpectrogram(call, handles, true, [], audioReader);
        call_spectrograms{i} = I;
    catch ME
        fprintf('Warning: Could not create spectrogram for call %d: %s\n', i, ME.message);
        call_spectrograms{i} = [];
    end
end

% Remove calls with failed spectrograms
valid_calls = ~cellfun(@isempty, call_spectrograms);
Calls = Calls(valid_calls, :);
call_spectrograms = call_spectrograms(valid_calls);

fprintf('Successfully created spectrograms for %d calls\n', sum(valid_calls));

%% Prepare Clustering Data
fprintf('Preparing clustering data...\n');

% Prepare clustering data structure
ClusteringData = table();
ClusteringData.Spectrogram = call_spectrograms;
ClusteringData.Box = call_boxes(valid_calls, :);
ClusteringData.Accept = ones(sum(valid_calls), 1);

% Add frequency and time contours if available
if isfield(Calls, 'xFreq') && ~isempty(Calls.xFreq)
    ClusteringData.xFreq = Calls.xFreq(valid_calls);
end
if isfield(Calls, 'xTime') && ~isempty(Calls.xTime)
    ClusteringData.xTime = Calls.xTime(valid_calls);
end

%% Clustering
fprintf('Performing clustering...\n');

if strcmp(clustering_method, 'VAE') && exist(custom_network_path, 'file')
    % Use custom VAE network for clustering
    fprintf('Using custom VAE network for clustering...\n');
    
    try
        custom_network = load(custom_network_path);
        
        % Extract features using VAE encoder
        num_calls = height(ClusteringData);
        call_spectrograms = ClusteringData.Spectrogram;
        
        % Get network options
        options = custom_network.options;
        encoderNet = custom_network.encoderNet;
        
                 % Initialize feature matrix - get dimensions from your network
         % Your encoder outputs 32-dimensional features, but cluster centers expect 48D
         encoder_output_dim = 32; % What your encoder produces
         cluster_center_dim = 48; % What your cluster centers expect
         cluster_features = zeros(num_calls, encoder_output_dim);
        
        % Process each call spectrogram
        for i = 1:num_calls
            spec = call_spectrograms{i};
            if isempty(spec)
                continue;
            end
            
            try
                % Preprocess spectrogram for VAE network
                processed_spec = preprocess_for_vae(spec, options);
                
                % Extract features using VAE encoder
                features = extract_vae_features(processed_spec, encoderNet);
                
                                 % Store features
                 if length(features) == encoder_output_dim
                     cluster_features(i, :) = features;
                 end
                
            catch ME
                fprintf('Warning: Failed to extract features for call %d: %s\n', i, ME.message);
                cluster_features(i, :) = 0;
            end
        end
        
                 % Perform clustering using pre-trained cluster centers
         valid_features = any(cluster_features ~= 0, 2);
         valid_calls = find(valid_features);
         valid_features_matrix = cluster_features(valid_features, :);
         
         % Combine VAE features with frequency contour features (DeepSqueak approach)
         % Extract frequency contours (16 dimensions like DeepSqueak does)
         freq_features = zeros(size(valid_features_matrix, 1), 16);
         for i = 1:size(valid_features_matrix, 1)
             call_idx = valid_calls(i);
             if isfield(ClusteringData, 'xFreq') && ~isempty(ClusteringData.xFreq{call_idx})
                 % Resize frequency contour to 16 dimensions (like DeepSqueak)
                 freq_contour = ClusteringData.xFreq{call_idx};
                 freq_features(i, :) = imresize(freq_contour', [1, 16]);
             else
                 % If no frequency contour, use zeros
                 freq_features(i, :) = 0;
             end
         end
         
         % Apply z-score normalization to both features (exactly like DeepSqueak)
         freq_features = zscore(freq_features, 0, 'all');
         valid_features_matrix = zscore(valid_features_matrix, 0, 'all');
         
         % Combine VAE features (32D) + frequency features (16D) = 48D
         combined_features = [valid_features_matrix, freq_features];
         valid_features_matrix = combined_features;
        
        if ~isempty(valid_features_matrix)
            % Use pre-trained cluster centers
            C = custom_network.C;
            clusterNames = custom_network.clusterName;
            
            % Find nearest cluster for each call
            [cluster_assignments_raw, distances] = knnsearch(C, valid_features_matrix, 'Distance', 'euclidean');
            
            % Map cluster assignments back to original call indices
            cluster_assignments = zeros(num_calls, 1);
            cluster_assignments(valid_calls) = cluster_assignments_raw;
            cluster_assignments(~valid_features) = -1;
            
            % Add cluster assignments to data
            ClusteringData.Cluster = cluster_assignments;
            ClusteringData.ClusterDistance = zeros(num_calls, 1);
            ClusteringData.ClusterDistance(valid_calls) = distances;
            
                         fprintf('VAE clustering completed. Found %d clusters\n', size(C, 1));
             
             % Export cluster results to CSV
             export_cluster_results_to_csv(audio_file, cluster_assignments, clusterNames, output_folder);
        else
            fprintf('No valid features extracted. Using contour-based clustering.\n');
            clustering_method = 'contour';
        end
        
    catch ME
        fprintf('VAE clustering failed: %s\n', ME.message);
        fprintf('Using contour-based clustering instead.\n');
        clustering_method = 'contour';
    end
end

if strcmp(clustering_method, 'contour')
    fprintf('Using contour-based clustering...\n');
    
    % Create simple feature matrix from spectrograms
    num_calls = height(ClusteringData);
    features = zeros(num_calls, 3); % [duration, bandwidth, power]
    
    for i = 1:num_calls
        spec = ClusteringData.Spectrogram{i};
        if ~isempty(spec)
            features(i, 1) = size(spec, 2);  % Duration
            features(i, 2) = size(spec, 1);  % Bandwidth
            features(i, 3) = mean(spec(:));  % Power
        end
    end
    
    % Normalize features
    features = zscore(features);
    
    % Simple k-means clustering
    num_clusters = min(5, num_calls);
    [cluster_assignments, cluster_centers] = kmeans(features, num_clusters, 'Replicates', 5);
    
    % Add cluster assignments to data
    ClusteringData.Cluster = cluster_assignments;
    
    fprintf('Contour clustering completed. Found %d clusters\n', num_clusters);
end

%% Save Results
if save_results
    % Create filename for this audio file
    [~, audio_name, ~] = fileparts(audio_file);
    
    % Save detection results
    detection_file = fullfile(output_folder, sprintf('%s_detected_calls.mat', audio_name));
    save(detection_file, 'Calls', 'audio_info', 'network_data', 'detection_time');
    
    % Save clustering data
    clustering_data_file = fullfile(output_folder, sprintf('%s_clustering_data.mat', audio_name));
    save(clustering_data_file, 'ClusteringData');
    
    % Save final results
    results_file = fullfile(output_folder, sprintf('%s_final_results.mat', audio_name));
    if exist('cluster_assignments', 'var')
        save(results_file, 'Calls', 'ClusteringData', 'call_spectrograms', 'cluster_assignments');
    else
        save(results_file, 'Calls', 'ClusteringData', 'call_spectrograms');
    end
    
    % Create summary report
    report_file = fullfile(output_folder, sprintf('%s_report.txt', audio_name));
    fid = fopen(report_file, 'w');
    fprintf(fid, 'DeepSqueak Processing Report\n');
    fprintf(fid, '==========================\n\n');
    fprintf(fid, 'Audio file: %s\n', audio_file);
    fprintf(fid, 'Network used: %s\n', network_file);
    fprintf(fid, 'Detection time: %.2f seconds\n', detection_time);
    fprintf(fid, 'Total calls detected: %d\n', height(Calls));
    fprintf(fid, 'Valid spectrograms: %d\n', sum(valid_calls));
    fprintf(fid, 'Clustering method: %s\n', clustering_method);
    if exist('cluster_assignments', 'var')
        fprintf(fid, 'Number of clusters: %d\n', length(unique(cluster_assignments(cluster_assignments > 0))));
    end
    fclose(fid);
    
    fprintf('Results saved to: %s\n', output_folder);
end

end

%% CSV Export Function
function export_cluster_results_to_csv(audio_file, cluster_assignments, clusterNames, output_folder)
    % Export cluster results to CSV format
    % This function creates a CSV with audio filename and cluster counts
    
    try
        % Get audio filename without path
        [~, audio_name, ~] = fileparts(audio_file);
        
        % Count calls in each cluster
        cluster_counts = zeros(length(clusterNames), 1);
        for i = 1:length(clusterNames)
            cluster_counts(i) = sum(cluster_assignments == i);
        end
        
        % Create CSV data
        csv_data = table();
        csv_data.AudioFile = {audio_name};
        
        % Add cluster counts with cluster names as column headers
        for i = 1:length(clusterNames)
            cluster_name = string(clusterNames(i));
            if isempty(cluster_name) || strcmp(cluster_name, '')
                cluster_name = sprintf('Cluster_%d', i);
            end
            csv_data.(cluster_name) = cluster_counts(i);
        end
        
        % Add total calls column
        csv_data.TotalCalls = sum(cluster_counts);
        
        % Add failed calls column
        failed_calls = sum(cluster_assignments == -1);
        if failed_calls > 0
            csv_data.FailedCalls = failed_calls;
        end
        
        % Save CSV file
        csv_filename = fullfile(output_folder, sprintf('%s_cluster_results.csv', audio_name));
        writetable(csv_data, csv_filename);
        fprintf('Cluster results exported to: %s\n', csv_filename);
        
        % Display CSV contents
        fprintf('\nCSV Export Summary:\n');
        fprintf('Audio File: %s\n', audio_name);
        for i = 1:length(clusterNames)
            cluster_name = string(clusterNames(i));
            if isempty(cluster_name) || strcmp(cluster_name, '')
                cluster_name = sprintf('Cluster_%d', i);
            end
            fprintf('%s: %d calls\n', cluster_name, cluster_counts(i));
        end
        fprintf('Total Calls: %d\n', sum(cluster_counts));
        if failed_calls > 0
            fprintf('Failed Calls: %d\n', failed_calls);
        end
        
    catch ME
        fprintf('Warning: Could not export CSV results: %s\n', ME.message);
    end
end

%% Helper Functions for VAE Processing

function processed_spec = preprocess_for_vae(spec, options)
%% Preprocess spectrogram for VAE network input
% Uses the exact same preprocessing as DeepSqueak

% Get target image size from network options
if isfield(options, 'imageSize')
    target_size = options.imageSize(1:2);
else
    target_size = [128, 128]; % DeepSqueak default
end

% Resize spectrogram to match network input size (like DeepSqueak)
processed_spec = imresize(spec, target_size);

% Convert to single precision and normalize by 256 (exactly like DeepSqueak)
processed_spec = single(processed_spec) ./ 256;

% Add batch dimension for network input
if ndims(processed_spec) == 2
    processed_spec = reshape(processed_spec, [size(processed_spec), 1]);
end

end

function features = extract_vae_features(processed_spec, encoderNet)
%% Extract features using VAE encoder network

try
    % Convert to dlarray for Deep Learning Toolbox
    if ~isa(processed_spec, 'dlarray')
        input_data = dlarray(processed_spec, 'SSCB');
    else
        input_data = processed_spec;
    end
    
    % Input is already normalized by 256 in preprocessing
    
    % Extract features using the encoder
    [~, zMean] = sampling(encoderNet, single(input_data));
    
    % Convert to regular array and ensure it's a row vector
    features = stripdims(zMean)';
    features = gather(extractdata(features));
    features = features(:)';
    
catch ME
    fprintf('Error in VAE feature extraction: %s\n', ME.message);
    rethrow(ME);
end

end
