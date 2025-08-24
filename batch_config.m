%% Batch Processing Configuration File
% Edit this file to customize your batch processing settings
% Then run batch_process_folders.m

%% Audio Directory Configuration
% Set the base directory containing your audio folders
% You can use relative or absolute paths
base_directory = 'Audio/';  % Change this to your audio base directory

% Set the output base directory for results
output_base_directory = 'batch_output/';

%% Detection Settings
% These are the same settings used in the DeepSqueak GUI
detection_settings = {
    '0',      % Detection length (0 = full file, or specify duration in seconds)
    '100',    % High frequency cutoff (kHz)
    '18',     % Low frequency cutoff (kHz)
    '0.5',   % Score cutoff (minimum confidence)
    '1'       % Enable detection (1 = enabled, 0 = disabled)
};

%% Network Configuration
% Choose which detection network to use
% Available networks in your Networks/ folder:
network_file = 'Networks/2025.02.18.Mouse.YoloR3.mat';  % Mouse detector
% network_file = 'Networks/2025.02.05.Rat.Long.YoloR2.mat';  % Long rat detector
% network_file = 'Networks/2025.02.04.Rat.YoloR2.mat';  % Rat detector

%% Clustering Configuration
% Choose clustering method
clustering_method = 'VAE';  % 'VAE' for your custom network, 'contour' for feature-based

% Path to your custom clustering network
custom_network_path = 'NNTest#1.mat';

%% Output Settings
% Enable/disable result saving
save_results = true;

% File formats to process (uncomment the ones you want)
audio_extensions = {
    '*.wav',   % WAV files
    '*.flac',  % FLAC files
    '*.mp3',   % MP3 files
    '*.m4a',   % M4A files
    '*.aiff',  % AIFF files
    '*.ogg'    % OGG files
};

%% Processing Options
% Enable parallel processing for faster execution (requires Parallel Computing Toolbox)
use_parallel = false;

% Maximum number of files to process per folder (set to 0 for unlimited)
max_files_per_folder = 0;

% Skip files that already have output (useful for resuming interrupted processing)
skip_existing_output = false;

% Create detailed logs
create_detailed_logs = true;

%% Advanced Settings
% Memory management for large files
max_audio_duration = 300;  % Maximum audio duration to process (seconds, 0 = unlimited)

% Spectrogram settings (if you need to override DeepSqueak defaults)
spectrogram_settings = struct();
spectrogram_settings.windowsize = 0.0032;  % Window size in seconds
spectrogram_settings.noverlap = 0.0016;    % Overlap in seconds
spectrogram_settings.nfft = 0.0032;        % FFT size in seconds

%% Example Configurations for Different Use Cases

% For mouse vocalization analysis:
% clustering_method = 'VAE';
% network_file = 'Networks/2025.02.18.Mouse.YoloR3.mat';
% detection_settings{2} = '100';  % High freq cutoff
% detection_settings{3} = '18';   % Low freq cutoff

% For rat vocalization analysis:
% clustering_method = 'VAE';
% network_file = 'Networks/2025.02.04.Rat.YoloR2.mat';
% detection_settings{2} = '80';   % High freq cutoff
% detection_settings{3} = '15';   % Low freq cutoff

% For long duration recordings:
% detection_settings{1} = '60';   % Process first 60 seconds
% max_audio_duration = 60;

% For high-quality analysis:
% detection_settings{4} = '0.7';  % Higher score threshold
% create_detailed_logs = true;
