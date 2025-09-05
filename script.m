%% DeepSqueak Audio Processing Script (No GUI)
% This script processes audio files using DeepSqueak's core functions
% without requiring the GUI interface.

clear; clc; close all;

%% Configuration
% Set paths and parameters
audio_file = 'FILEPATH';  % Replace with your audio file path
network_file = 'Networks/2025.02.18.Mouse.YoloR3.mat';  % Choose your network
output_folder = 'output/';  % Output folder for results

% Detection settings (same as DeepSqueak GUI)
detection_settings = {
    '0',      % Detection length (0 = full file)
    '100',    % High frequency cutoff (kHz)
    '18',     % Low frequency cutoff (kHz)
    '0.5',   % Score cutoff
    '1'       % Enable detection
};

% Clustering settings
clustering_method = 'VAE';  % 'VAE' or 'contour'
save_results = true;

%% Initialize
fprintf('Initializing DeepSqueak processing...\n');

% Create output directory
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

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
%fprintf('Sample rate: %d Hz\n', audio_info.SampleRate);
%fprintf('Channels: %d\n', audio_info.NumChannels);

%% Step 1: Load Neural Network
fprintf('\n=== Step 1: Loading Neural Network ===\n');
%fprintf('Looking for network file: %s\n', network_file);
if ~exist(network_file, 'file')
    error('Network file not found: %s', network_file);
end

fprintf('Loading network file...\n');
network_data = load(network_file);
%fprintf('Network file loaded successfully\n');
fprintf('Network file: %s\n', network_file);
fprintf('Network type: %s\n', class(network_data.detector));
fprintf('Network variables: %s\n', strjoin(fieldnames(network_data), ', '));

%% Step 2: Detect Calls Using Neural Network
fprintf('\n=== Step 2: Detecting Calls ===\n');
fprintf('Detection settings:\n');
fprintf('  - Detection length: %s (0 = full file)\n', detection_settings{1});
fprintf('  - High frequency cutoff: %s kHz\n', detection_settings{2});
fprintf('  - Low frequency cutoff: %s kHz\n', detection_settings{3});
fprintf('  - Score cutoff: %s\n', detection_settings{4});
fprintf('  - Enable detection: %s\n', detection_settings{5});

% Convert settings to numeric values
%fprintf('Converting settings to numeric values...\n');
settings = zeros(1, length(detection_settings));
for i = 1:length(detection_settings)
    settings(i) = str2double(detection_settings{i});
end

% Detect calls using SqueakDetect function
%fprintf('Running call detection with SqueakDetect...\n');
%fprintf('This may take a while depending on audio file length...\n');
tic;
Calls = SqueakDetect(audio_file, network_data, 'Audio File', settings, 1, 1, 'Detection Network');
detection_time = toc;

fprintf('Detection completed in %.2f seconds\n', detection_time);
fprintf('Found %d calls\n', height(Calls));

if isempty(Calls)
    fprintf('No calls detected. Exiting.\n');
    return;
end

% Save detection results
if save_results
    detection_file = fullfile(output_folder, 'detected_calls.mat');
    save(detection_file, 'Calls', 'audio_info', 'network_data', 'detection_time');
    fprintf('Detection results saved to: %s\n', detection_file);
end

%% Step 3: Create Spectrograms for Each Call
fprintf('\n=== Step 3: Creating Call Spectrograms ===\n');

% Initialize audio reader
audioReader = squeakData([]);
audioReader.audiodata = audio_info;

% Create spectrograms for each detected call
call_spectrograms = cell(height(Calls), 1);
call_boxes = Calls.Box;

fprintf('Creating spectrograms for %d calls...\n', height(Calls));

for i = 1:height(Calls)
    if mod(i, 10) == 0
        fprintf('Processing call %d/%d...\n', i, height(Calls));
    end
    
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

%% Step 4: Extract Call Images and Features
fprintf('\n=== Step 4: Extracting Call Images ===\n');

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

% Save clustering data
if save_results
    clustering_data_file = fullfile(output_folder, 'clustering_data.mat');
    save(clustering_data_file, 'ClusteringData');
    fprintf('Clustering data saved to: %s\n', clustering_data_file);
end

%% Step 5: Clustering Calls
fprintf('\n=== Step 5: Clustering Calls ===\n');

switch clustering_method
    case 'VAE'
        fprintf('Using VAE-based clustering...\n');
        
        % Check if VAE model exists
        if exist('Functions/Variational Autoencoder/create_VAE_model.m', 'file')
            try
                % Create VAE model (this would normally be done through GUI)
                % For now, we'll use a simplified approach
                fprintf('VAE clustering requires pre-trained model. Using contour-based clustering instead.\n');
                clustering_method = 'contour';
            catch ME
                fprintf('VAE clustering failed: %s\n', ME.message);
                fprintf('Falling back to contour-based clustering...\n');
                clustering_method = 'contour';
            end
        else
            fprintf('VAE functions not found. Using contour-based clustering...\n');
            clustering_method = 'contour';
        end
end

if strcmp(clustering_method, 'contour')
    fprintf('Using contour-based clustering...\n');
    
    % Extract contour features for clustering
    % This is a simplified version - in practice you'd use the full CreateClusteringData function
    
    % Create simple feature matrix from spectrograms
    num_calls = height(ClusteringData);
    features = zeros(num_calls, 3); % [duration, bandwidth, power]
    
    for i = 1:num_calls
        spec = ClusteringData.Spectrogram{i};
        if ~isempty(spec)
            % Duration (time dimension)
            features(i, 1) = size(spec, 2);
            % Bandwidth (frequency dimension)
            features(i, 2) = size(spec, 1);
            % Power (mean intensity)
            features(i, 3) = mean(spec(:));
        end
    end
    
    % Normalize features
    features = zscore(features);
    
    % Simple k-means clustering
    num_clusters = min(5, num_calls); % Adjust number of clusters as needed
    [cluster_assignments, cluster_centers] = kmeans(features, num_clusters, 'Replicates', 5);
    
    % Add cluster assignments to data
    ClusteringData.Cluster = cluster_assignments;
    
    fprintf('Clustering completed. Found %d clusters.\n', num_clusters);
    
    % Display cluster statistics
    for i = 1:num_clusters
        cluster_size = sum(cluster_assignments == i);
        fprintf('Cluster %d: %d calls (%.1f%%)\n', i, cluster_size, 100*cluster_size/num_calls);
    end
end

%% Step 6: Save Results and Create Visualizations
fprintf('\n=== Step 6: Saving Results ===\n');

if save_results
    % Save final results
    results_file = fullfile(output_folder, 'final_results.mat');
    save(results_file, 'Calls', 'ClusteringData', 'call_spectrograms', 'cluster_assignments', 'cluster_centers');
    fprintf('Final results saved to: %s\n', results_file);
    
    % Create summary report
    report_file = fullfile(output_folder, 'processing_report.txt');
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
        fprintf(fid, 'Number of clusters: %d\n', length(unique(cluster_assignments)));
    end
    fclose(fid);
    
    fprintf('Processing report saved to: %s\n', report_file);
end

%% Step 7: Load Your Custom Neural Network for Clustering
fprintf('\n=== Step 7: Custom Neural Network Integration ===\n');

% Load your trained clustering network
custom_network_path = 'NN FILE PATH';  % Your trained network FILE PATH

if exist(custom_network_path, 'file')
    fprintf('Loading custom clustering network: %s\n', custom_network_path);
    custom_network = load(custom_network_path);
    
    % Display network information
    fprintf('Network loaded successfully.\n');
    fprintf('Network variables: %s\n', strjoin(fieldnames(custom_network), ', '));
    
    % Use your custom VAE network for clustering
    fprintf('Using VAE-based clustering with your trained network...\n');
    
    try
        % Extract features using your VAE encoder
        num_calls = height(ClusteringData);
        call_spectrograms = ClusteringData.Spectrogram;
        
        % Get network options
        fprintf('Extracting network configuration...\n');
        options = custom_network.options;
        encoderNet = custom_network.encoderNet;
        
        fprintf('VAE network configuration loaded\n');
        fprintf('  - Input image size: %s\n', mat2str(options.imageSize));
        fprintf('  - Frequency range: %s kHz\n', mat2str(options.freqRange));
        fprintf('  - Max duration: %.3f seconds\n', options.maxDuration);
        
        % Validate network structure
        if ~isa(encoderNet, 'dlnetwork')
            error('Encoder network is not a valid dlnetwork object');
        end
        fprintf('Encoder network validation passed\n');
        
        % Initialize feature matrix - get dimensions from your network
        % Your encoder outputs 32-dimensional features, but cluster centers expect 48D
        encoder_output_dim = 32; % What your encoder produces
        cluster_center_dim = 48; % What your cluster centers expect
        cluster_features = zeros(num_calls, encoder_output_dim);
        
        % Process each call spectrogram
        fprintf('Extracting VAE features for %d calls...\n', num_calls);
        
        for i = 1:num_calls
            if mod(i, 10) == 0
                fprintf('Processing call %d/%d...\n', i, num_calls);
            end
            
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
                else
                    fprintf('Warning: Call %d features have unexpected dimensions: %d\n', i, length(features));
                end
                
            catch ME
                fprintf('Warning: Failed to extract features for call %d: %s\n', i, ME.message);
                cluster_features(i, :) = 0;
            end
        end
        
        % Perform clustering using your pre-trained cluster centers
        fprintf('Performing clustering using pre-trained cluster centers...\n');
        
        % Remove calls with zero features
        valid_features = any(cluster_features ~= 0, 2);
        valid_calls = find(valid_features);
        valid_features_matrix = cluster_features(valid_features, :);
        
        % Combine VAE features with frequency contour features (DeepSqueak approach)
        fprintf('Combining VAE features with frequency contour features...\n');
        
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
        fprintf('Applying z-score normalization to features...\n');
        freq_features = zscore(freq_features, 0, 'all');
        valid_features_matrix = zscore(valid_features_matrix, 0, 'all');
        
        % Combine VAE features (32D) + frequency features (16D) = 48D
        combined_features = [valid_features_matrix, freq_features];
        fprintf('Combined features: %dD VAE + %dD frequency = %dD total (z-score normalized)\n', ...
            size(valid_features_matrix, 2), size(freq_features, 2), size(combined_features, 2));
        
        valid_features_matrix = combined_features;
        
        if ~isempty(valid_features_matrix)
            % Use your pre-trained cluster centers
            C = custom_network.C;
            clusterNames = custom_network.clusterName;
            
            fprintf('Using %d pre-trained cluster centers\n', size(C, 1));
            fprintf('Cluster names: %s\n', strjoin(string(clusterNames), ', '));
            
            % Find nearest cluster for each call
            [cluster_assignments_raw, distances] = knnsearch(C, valid_features_matrix, 'Distance', 'euclidean');
            
            % Map cluster assignments back to original call indices
            cluster_assignments = zeros(num_calls, 1);
            cluster_assignments(valid_calls) = cluster_assignments_raw;
            cluster_assignments(~valid_features) = -1; % -1 indicates failed processing
            
            % Add cluster assignments to data
            ClusteringData.Cluster = cluster_assignments;
            ClusteringData.ClusterDistance = zeros(num_calls, 1);
            ClusteringData.ClusterDistance(valid_calls) = distances;
            
            fprintf('VAE clustering completed successfully!\n');
            fprintf('Number of clusters: %d\n', size(C, 1));
            
            % Display cluster statistics
            for i = 1:size(C, 1)
                cluster_size = sum(cluster_assignments == i);
                if cluster_size > 0
                    fprintf('Cluster %d (%s): %d calls (%.1f%%)\n', i, string(clusterNames(i)), cluster_size, 100*cluster_size/sum(valid_features));
                end
            end
            
            if sum(cluster_assignments == -1) > 0
                fprintf('Failed calls: %d (%.1f%%)\n', sum(cluster_assignments == -1), 100*sum(cluster_assignments == -1)/num_calls);
            end
            
            % Export cluster results to CSV
            fprintf('\n=== EXPORTING CLUSTER RESULTS ===\n');
            export_cluster_results_to_csv(audio_file, cluster_assignments, clusterNames, output_folder);
            
        else
            fprintf('No valid features extracted. Falling back to contour-based clustering.\n');
            clustering_method = 'contour';
        end
        
    catch ME
        fprintf('VAE clustering failed: %s\n', ME.message);
        fprintf('Falling back to contour-based clustering...\n');
        clustering_method = 'contour';
    end
    
else
    fprintf('Custom clustering network not found at: %s\n', custom_network_path);
    fprintf('Using contour-based clustering instead.\n');
    clustering_method = 'contour';
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

%% Summary
fprintf('\n=== Processing Complete ===\n');
fprintf('Successfully processed audio file: %s\n', audio_file);
fprintf('Detected calls: %d\n', height(Calls));
fprintf('Output folder: %s\n', output_folder);

if exist('cluster_assignments', 'var')
    fprintf('Clustering completed with %d clusters\n', length(unique(cluster_assignments)));
end

fprintf('\nScript completed successfully!\n');

%% Helper Functions for VAE Processing

function processed_spec = preprocess_for_vae(spec, options)
%% Preprocess spectrogram for VAE network input
% This function prepares spectrograms for your trained VAE network
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
% This function uses your trained VAE encoder to extract features

try
    % Convert to dlarray for Deep Learning Toolbox
    if ~isa(processed_spec, 'dlarray')
        input_data = dlarray(processed_spec, 'SSCB');
    else
        input_data = processed_spec;
    end
    
    % Input is already normalized by 256 in preprocessing
    
    % Extract features using the encoder
    % The encoder should output the latent representation
    [~, zMean] = sampling(encoderNet, single(input_data));
    
    % Convert to regular array and ensure it's a row vector
    features = stripdims(zMean)';
    features = gather(extractdata(features));
    features = features(:)';
    % Test
catch ME
    fprintf('Error in VAE feature extraction: %s\n', ME.message);
    rethrow(ME);
end

end
