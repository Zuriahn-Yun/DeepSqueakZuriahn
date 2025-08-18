function [calls, audioReader] = wav_to_call_detection(wav_file_path, detection_options)
% WAV_TO_CALL_DETECTION - Standalone function to detect calls from WAV file
% 
% Inputs:
%   wav_file_path - (optional) Path to the .wav file. If empty, opens file picker
%   detection_options - (optional) Structure with detection parameters
%
% Outputs:
%   calls - Structure array with detected call information
%   audioReader - Audio data structure for further processing

% Handle file selection
if nargin < 1 || isempty(wav_file_path)
    [filename, pathname] = uigetfile({'*.wav;*.WAV', 'WAV Files (*.wav)'; ...
                                     '*.*', 'All Files (*.*)'}, ...
                                     'Select WAV file for call detection');
    if filename == 0
        fprintf('No file selected. Exiting.\n');
        calls = [];
        audioReader = [];
        return;
    end
    wav_file_path = fullfile(pathname, filename);
end

if nargin < 2 || isempty(detection_options)
    % Default detection parameters - adjust these based on your needs
    detection_options = struct();
    detection_options.threshold = 0.1;           % Detection threshold
    detection_options.min_duration = 0.003;     % Minimum call duration (3ms)
    detection_options.max_duration = 2.0;       % Maximum call duration (2s)
    detection_options.freq_range = [15, 100];   % Frequency range in kHz
    detection_options.merge_gap = 0.01;         % Merge calls closer than 10ms
    detection_options.smoothing_window = 0.002; % 2ms smoothing window
end

%% Load audio file
fprintf('Loading audio file: %s\n', wav_file_path);
[audio_samples, sample_rate] = audioread(wav_file_path);

% Convert to mono if stereo
if size(audio_samples, 2) > 1
    audio_samples = mean(audio_samples, 2);
end

% Create audioReader structure (mimicking the GUI structure)
audioReader = struct();
audioReader.audiodata.SampleRate = sample_rate;
audioReader.AudioSamples = @(start_time, end_time) get_audio_segment(audio_samples, sample_rate, start_time, end_time);

%% Create spectrogram for detection
fprintf('Creating spectrogram for detection...\n');
% Use detection-appropriate spectrogram parameters
window_size = round(sample_rate * 0.002); % 2ms window
overlap = round(window_size * 0.8);       % 80% overlap
nfft = round(sample_rate * 0.004);        % 4ms FFT

[S, F, T] = batch_process_wav_files(audio_samples, window_size, overlap, nfft, sample_rate);
S_mag = abs(S);

% Convert frequency to kHz for easier handling
F_khz = F / 1000;

%% Frequency filtering
freq_mask = F_khz >= detection_options.freq_range(1) & F_khz <= detection_options.freq_range(2);
S_filtered = S_mag(freq_mask, :);
F_filtered = F_khz(freq_mask);

%% Detection algorithm
fprintf('Detecting calls...\n');
% Calculate energy in frequency band
energy_profile = mean(S_filtered, 1);

% Smooth the energy profile
smooth_samples = round(detection_options.smoothing_window * sample_rate / (window_size - overlap));
if smooth_samples > 1
    energy_profile = movmean(energy_profile, smooth_samples);
end

% Threshold detection
threshold = detection_options.threshold * max(energy_profile);
above_threshold = energy_profile > threshold;

% Find continuous segments above threshold
diff_above = diff([false, above_threshold, false]);
call_starts = find(diff_above == 1);
call_ends = find(diff_above == -1) - 1;

%% Process detected segments
calls = struct();
call_count = 0;

for i = 1:length(call_starts)
    start_idx = call_starts(i);
    end_idx = call_ends(i);
    
    % Convert to time
    start_time = T(start_idx);
    end_time = T(end_idx);
    duration = end_time - start_time;
    
    % Filter by duration
    if duration < detection_options.min_duration || duration > detection_options.max_duration
        continue;
    end
    
    % Find frequency bounds for this call segment
    call_segment = S_filtered(:, start_idx:end_idx);
    freq_energy = mean(call_segment, 2);
    
    % Find frequency range containing significant energy (e.g., above 10% of max)
    energy_threshold = 0.1 * max(freq_energy);
    freq_indices = find(freq_energy > energy_threshold);
    
    if isempty(freq_indices)
        continue;
    end
    
    min_freq = F_filtered(freq_indices(1));
    max_freq = F_filtered(freq_indices(end));
    
    % Create call structure
    call_count = call_count + 1;
    
    % Box format: [start_time, min_freq, duration, freq_range]
    calls(call_count).Box = [start_time, min_freq, duration, max_freq - min_freq];
    calls(call_count).RelBox = calls(call_count).Box; % For compatibility
    calls(call_count).Rate = sample_rate;
    calls(call_count).Type = 'USV'; % Default type
    calls(call_count).Score = mean(mean(call_segment)); % Average energy as score
    
    % Extract audio for this call (with some padding)
    padding = 0.01; % 10ms padding
    audio_start = max(1, round((start_time - padding) * sample_rate));
    audio_end = min(length(audio_samples), round((end_time + padding) * sample_rate));
    calls(call_count).Audio = {audio_samples(audio_start:audio_end)};
    
    fprintf('Call %d: %.3fs-%.3fs, %.1f-%.1fkHz, Score: %.3f\n', ...
        call_count, start_time, end_time, min_freq, max_freq, calls(call_count).Score);
end

%% Merge nearby calls
if ~isempty(calls) && detection_options.merge_gap > 0
    calls = merge_nearby_calls(calls, detection_options.merge_gap);
end

fprintf('Detection complete. Found %d calls.\n', length(calls));

%% Save results
[filepath, name, ~] = fileparts(wav_file_path);
output_file = fullfile(filepath, [name '_calls.mat']);
save(output_file, 'calls', 'audioReader', 'detection_options');
fprintf('Saved results to: %s\n', output_file);

end

%% Helper function to extract audio segments
function audio_segment = get_audio_segment(audio_samples, sample_rate, start_time, end_time)
    start_idx = max(1, round(start_time * sample_rate));
    end_idx = min(length(audio_samples), round(end_time * sample_rate));
    audio_segment = audio_samples(start_idx:end_idx);
end

%% Helper function to merge nearby calls
function merged_calls = merge_nearby_calls(calls, merge_gap)
    if length(calls) <= 1
        merged_calls = calls;
        return;
    end
    
    % Sort calls by start time
    start_times = arrayfun(@(x) x.Box(1), calls);
    [~, sort_idx] = sort(start_times);
    calls = calls(sort_idx);
    
    merged_calls = calls(1);
    merged_idx = 1;
    
    for i = 2:length(calls)
        current_call = calls(i);
        last_merged = merged_calls(merged_idx);
        
        % Check if calls should be merged
        gap = current_call.Box(1) - (last_merged.Box(1) + last_merged.Box(3));
        
        if gap <= merge_gap
            % Merge calls
            new_start = min(last_merged.Box(1), current_call.Box(1));
            new_end = max(last_merged.Box(1) + last_merged.Box(3), ...
                         current_call.Box(1) + current_call.Box(3));
            new_min_freq = min(last_merged.Box(2), current_call.Box(2));
            new_max_freq = max(last_merged.Box(2) + last_merged.Box(4), ...
                              current_call.Box(2) + current_call.Box(4));
            
            merged_calls(merged_idx).Box = [new_start, new_min_freq, ...
                                          new_end - new_start, new_max_freq - new_min_freq];
            merged_calls(merged_idx).RelBox = merged_calls(merged_idx).Box;
            merged_calls(merged_idx).Score = max(last_merged.Score, current_call.Score);
            
            % Merge audio (this is simplified - in practice you'd want to extract the full range)
            fprintf('Merged calls at %.3fs and %.3fs\n', last_merged.Box(1), current_call.Box(1));
        else
            % Add as new call
            merged_idx = merged_idx + 1;
            merged_calls(merged_idx) = current_call;
        end
    end
    
    fprintf('Merged %d calls into %d calls\n', length(calls), length(merged_calls));
end

%% Example usage function
function example_usage()
    % Example 1: Let user pick file interactively
    [calls, audioReader] = wav_to_call_detection(); % Opens file picker
    
    % Example 2: With specific file and custom options
    options = struct();
    options.threshold = 0.15;           % Adjust based on your data
    options.min_duration = 0.005;      % 5ms minimum
    options.max_duration = 1.0;        % 1s maximum
    options.freq_range = [20, 80];     % 20-80 kHz
    options.merge_gap = 0.02;          % 20ms merge gap
    
    wav_file = 'path/to/your/file.wav';
    [calls, audioReader] = wav_to_call_detection(wav_file, options);
    
    % Now you can use calls with your CreateFocusSpectrogram function
    % for i = 1:length(calls)
    %     [I, windowsize, noverlap, nfft, rate, box, s, fr, ti, audio, p] = ...
    %         CreateFocusSpectrogram(calls(i), handles, true, options, audioReader);
    % end
end

%% Batch processing function
function batch_process_wav_files()
    % Process multiple WAV files at once
    [filenames, pathname] = uigetfile({'*.wav;*.WAV', 'WAV Files (*.wav)'}, ...
                                     'Select WAV files for batch processing', ...
                                     'MultiSelect', 'on');
    
    if iscell(filenames)
        % Multiple files selected
        fprintf('Processing %d files...\n', length(filenames));
        for i = 1:length(filenames)
            wav_file = fullfile(pathname, filenames{i});
            fprintf('\nProcessing file %d/%d: %s\n', i, length(filenames), filenames{i});
            [calls, audioReader] = wav_to_call_detection(wav_file);
        end
    elseif ischar(filenames)
        % Single file selected
        wav_file = fullfile(pathname, filenames);
        [calls, audioReader] = wav_to_call_detection(wav_file);
    else
        fprintf('No files selected.\n');
    end
end