%% Custom Neural Network Integration for DeepSqueak Clustering
% This script shows how to integrate your trained neural network for clustering
% with the DeepSqueak call detection pipeline.

function [cluster_assignments, cluster_features] = custom_clustering_integration(ClusteringData, custom_network_path)
%% Custom Clustering Integration
% Inputs:
%   ClusteringData - Table containing call spectrograms and metadata
%   custom_network_path - Path to your trained neural network .mat file
%
% Outputs:
%   cluster_assignments - Vector of cluster assignments for each call
%   cluster_features - Extracted features from your neural network

%% Load Your Custom Neural Network
fprintf('Loading custom clustering network: %s\n', custom_network_path);

if ~exist(custom_network_path, 'file')
    error('Custom network file not found: %s', custom_network_path);
end

% Load the network (adjust variable names based on your network structure)
network_data = load(custom_network_path);

% Display network information
fprintf('Network loaded successfully.\n');
fprintf('Network variables: %s\n', strjoin(fieldnames(network_data), ', '));

%% Extract Features Using Your Neural Network
fprintf('Extracting features using custom neural network...\n');

num_calls = height(ClusteringData);
call_spectrograms = ClusteringData.Spectrogram;

% Initialize feature matrix
% Adjust dimensions based on your network's output
feature_dimensions = 128; % Example: adjust to match your network's feature output
cluster_features = zeros(num_calls, feature_dimensions);

% Process each call spectrogram
for i = 1:num_calls
    if mod(i, 10) == 0
        fprintf('Processing call %d/%d...\n', i, num_calls);
    end
    
    spec = call_spectrograms{i};
    if isempty(spec)
        continue;
    end
    
    try
        % Preprocess spectrogram for your network
        % Adjust preprocessing based on your network's requirements
        processed_spec = preprocess_spectrogram_for_network(spec);
        
        % Extract features using your network
        % This is where you'd call your specific neural network
        features = extract_features_with_network(processed_spec, network_data);
        
        % Store features
        if length(features) == feature_dimensions
            cluster_features(i, :) = features;
        else
            fprintf('Warning: Call %d features have unexpected dimensions: %d\n', i, length(features));
        end
        
    catch ME
        fprintf('Warning: Failed to extract features for call %d: %s\n', i, ME.message);
        % Use zero features for failed extractions
        cluster_features(i, :) = 0;
    end
end

%% Perform Clustering on Extracted Features
fprintf('Performing clustering on extracted features...\n');

% Remove calls with zero features (failed extractions)
valid_features = any(cluster_features ~= 0, 2);
valid_calls = find(valid_features);
valid_features_matrix = cluster_features(valid_features, :);

if isempty(valid_features_matrix)
    error('No valid features extracted. Check your network and preprocessing.');
end

% Normalize features
valid_features_matrix = zscore(valid_features_matrix);

% Determine optimal number of clusters
% You can adjust this based on your domain knowledge
max_clusters = min(10, size(valid_features_matrix, 1));
optimal_clusters = estimate_optimal_clusters(valid_features_matrix, max_clusters);

fprintf('Estimated optimal number of clusters: %d\n', optimal_clusters);

% Perform clustering (you can use different methods)
% Option 1: K-means clustering
[cluster_assignments_raw, cluster_centers] = kmeans(valid_features_matrix, optimal_clusters, 'Replicates', 10);

% Option 2: Hierarchical clustering (uncomment if preferred)
% cluster_tree = linkage(valid_features_matrix, 'ward');
% cluster_assignments_raw = cluster(cluster_tree, optimal_clusters);

% Option 3: DBSCAN clustering (uncomment if preferred)
% epsilon = 0.5; % Adjust based on your feature space
% min_pts = 3;
% cluster_assignments_raw = dbscan(valid_features_matrix, epsilon, min_pts);

% Map cluster assignments back to original call indices
cluster_assignments = zeros(num_calls, 1);
cluster_assignments(valid_calls) = cluster_assignments_raw;

% Handle calls with failed feature extraction
cluster_assignments(~valid_features) = -1; % -1 indicates failed processing

%% Display Clustering Results
fprintf('\n=== Clustering Results ===\n');
fprintf('Total calls processed: %d\n', num_calls);
fprintf('Successful feature extractions: %d\n', sum(valid_features));
fprintf('Failed feature extractions: %d\n', sum(~valid_features));
fprintf('Number of clusters: %d\n', optimal_clusters);

% Display cluster statistics
for i = 1:optimal_clusters
    cluster_size = sum(cluster_assignments == i);
    if cluster_size > 0
        fprintf('Cluster %d: %d calls (%.1f%%)\n', i, cluster_size, 100*cluster_size/sum(valid_features));
    end
end

if sum(cluster_assignments == -1) > 0
    fprintf('Failed calls: %d (%.1f%%)\n', sum(cluster_assignments == -1), 100*sum(cluster_assignments == -1)/num_calls);
end

end

%% Helper Functions

function processed_spec = preprocess_spectrogram_for_network(spec)
%% Preprocess spectrogram for neural network input
% Adjust this function based on your network's input requirements

% Ensure spectrogram is the right size
% You might need to resize to match your network's expected input dimensions
target_size = [64, 64]; % Example: adjust to your network's input size

% Resize spectrogram
processed_spec = imresize(spec, target_size);

% Normalize to [0, 1] range
processed_spec = (processed_spec - min(processed_spec(:))) / (max(processed_spec(:)) - min(processed_spec(:)));

% Convert to single precision if needed
processed_spec = single(processed_spec);

% Add batch dimension if needed
if ndims(processed_spec) == 2
    processed_spec = reshape(processed_spec, [size(processed_spec), 1]);
end

end

function features = extract_features_with_network(processed_spec, network_data)
%% Extract features using your neural network
% This is where you'd implement the actual feature extraction
% Adjust based on your network's architecture and how it's saved

% Example implementation - modify based on your network structure
try
    % If your network is a MATLAB network object
    if isfield(network_data, 'net') && isa(network_data.net, 'network')
        net = network_data.net;
        % Convert input to network format if needed
        input_data = processed_spec;
        features = sim(net, input_data);
        features = features(:)'; % Ensure it's a row vector
        
    % If your network is a Deep Learning Toolbox network
    elseif isfield(network_data, 'net') && (isa(network_data.net, 'dlnetwork') || isa(network_data.net, 'SeriesNetwork'))
        net = network_data.net;
        % Convert to dlarray if needed
        if ~isa(processed_spec, 'dlarray')
            input_data = dlarray(processed_spec, 'SSCB');
        else
            input_data = processed_spec;
        end
        
        % Extract features from a specific layer (adjust layer name)
        % You might need to modify this based on your network architecture
        try
            % Try to get features from the last layer before classification
            features = predict(net, input_data);
        catch
            % Fallback: use the network output
            features = predict(net, input_data);
        end
        
        % Convert to regular array and ensure it's a row vector
        features = extractdata(features);
        features = features(:)';
        
    % If your network is saved as a different format
    elseif isfield(network_data, 'encoder') && isfield(network_data, 'options')
        % This might be a VAE or autoencoder
        encoder = network_data.encoder;
        options = network_data.options;
        
        % Resize to match expected input size
        if isfield(options, 'imageSize')
            processed_spec = imresize(processed_spec, options.imageSize(1:2));
        end
        
        % Convert to dlarray
        input_data = dlarray(single(processed_spec) ./ 256, 'SSCB');
        
        % Extract features using the encoder
        [~, zMean] = sampling(encoder, single(input_data));
        features = stripdims(zMean)';
        features = gather(extractdata(features));
        features = features(:)';
        
    else
        % Custom network format - implement based on your specific structure
        error('Unsupported network format. Please implement custom feature extraction.');
    end
    
catch ME
    fprintf('Error in feature extraction: %s\n', ME.message);
    fprintf('Network data fields: %s\n', strjoin(fieldnames(network_data), ', '));
    rethrow(ME);
end

end

function optimal_clusters = estimate_optimal_clusters(features, max_clusters)
%% Estimate optimal number of clusters using elbow method
% This is a simple heuristic - you can implement more sophisticated methods

if size(features, 1) < 3
    optimal_clusters = 1;
    return;
end

% Calculate within-cluster sum of squares for different numbers of clusters
wcss = zeros(max_clusters, 1);

for k = 1:max_clusters
    try
        [~, ~, sumd] = kmeans(features, k, 'Replicates', 5);
        wcss(k) = sum(sumd);
    catch
        wcss(k) = inf;
    end
end

% Find elbow point (simplified method)
if max_clusters > 2
    % Calculate second derivative
    second_derivative = diff(diff(wcss));
    [~, elbow_idx] = max(abs(second_derivative));
    optimal_clusters = elbow_idx + 1;
else
    optimal_clusters = 2;
end

% Ensure reasonable bounds
optimal_clusters = max(2, min(optimal_clusters, max_clusters));

end
