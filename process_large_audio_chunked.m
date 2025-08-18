function [combined_S, combined_F, combined_T] = process_large_audio_chunked(audio_file_path, chunk_duration_minutes)
% Automatically process large audio files in chunks and combine results
% This replaces your original wav_to_call_detection call
% 
% Inputs:
%   audio_file_path - path to your large audio file
%   chunk_duration_minutes - duration of each chunk in minutes (default: 5)
%
% Outputs:
%   combined_S - combined spectrogram matrix
%   combined_F - frequency vector
%   combined_T - time vector (adjusted for full file)

if nargin < 2
    chunk_duration_minutes = 5; % Default 5 minutes per chunk
end

fprintf('Loading large audio file: %s\n', audio_file_path);

% Load the audio file
try
    [audio_samples, sample_rate] = audioread(audio_file_path);
    fprintf('Audio loaded. Duration: %.2f minutes, Sample rate: %d Hz\n', ...
            length(audio_samples)/sample_rate/60, sample_rate);
catch ME
    error('Failed to load audio file: %s', ME.message);
end

% Convert to mono if stereo
if size(audio_samples, 2) > 1
    audio_samples = mean(audio_samples, 2);
    fprintf('Converted stereo to mono\n');
end

% Calculate chunk parameters
chunk_duration_seconds = chunk_duration_minutes * 60;
chunk_samples = round(chunk_duration_seconds * sample_rate);
total_samples = length(audio_samples);
num_chunks = ceil(total_samples / chunk_samples);

fprintf('Processing in %d chunks of %.1f minutes each...\n', num_chunks, chunk_duration_minutes);

% Spectrogram parameters (adjust these if still too memory intensive)
window_size = 1024;
overlap = 512;
nfft = 1024;

% Initialize storage for combined results
combined_S = [];
combined_T = [];
combined_F = [];
first_chunk = true;

% Process each chunk
for chunk_idx = 1:num_chunks
    fprintf('Processing chunk %d/%d... ', chunk_idx, num_chunks);
    
    % Calculate chunk boundaries
    start_idx = (chunk_idx - 1) * chunk_samples + 1;
    end_idx = min(start_idx + chunk_samples - 1, total_samples);
    
    % Extract chunk
    audio_chunk = audio_samples(start_idx:end_idx);
    
    % Calculate time offset for this chunk
    time_offset = (start_idx - 1) / sample_rate;
    
    try
        % Process chunk with spectrogram
        [S_chunk, F_chunk, T_chunk] = spectrogram(audio_chunk, window_size, overlap, nfft, sample_rate);
        
        % Adjust time vector to account for chunk position
        T_chunk_adjusted = T_chunk + time_offset;
        
        % Combine results
        if first_chunk
            combined_S = S_chunk;
            combined_F = F_chunk; % Frequency vector is the same for all chunks
            combined_T = T_chunk_adjusted;
            first_chunk = false;
        else
            % Concatenate spectrogram matrices along time dimension
            combined_S = [combined_S, S_chunk];
            combined_T = [combined_T, T_chunk_adjusted];
        end
        
        fprintf('Done (%.1fs processed)\n', length(audio_chunk)/sample_rate);
        
    catch ME
        fprintf('ERROR: %s\n', ME.message);
        % Continue with next chunk even if this one fails
        continue;
    end
end

fprintf('\nCombined spectrogram created successfully!\n');
fprintf('Final dimensions: %d frequencies x %d time points\n', size(combined_S, 1), size(combined_S, 2));
fprintf('Total time span: %.2f minutes\n', (combined_T(end) - combined_T(1))/60);

end

% Modified version of your original detection function that uses chunking
function detect_calls_large_file(audio_file_path)
% Drop-in replacement for wav_to_call_detection that handles large files
% This is what you actually call instead of your original function

fprintf('Starting call detection on large audio file...\n');

% Process the large audio file in chunks
[S, F, T] = process_large_audio_chunked(audio_file_path, 5); % 5-minute chunks

fprintf('Spectrogram processing complete. Now running call detection...\n');

% Now you can continue with your original DeepSqueak detection code
% using the combined S, F, T variables just like before

% Example of how to continue (replace with your actual detection code):
% detection_threshold = 0.5;  % Your detection parameters
% [calls, scores] = your_detection_function(S, F, T, detection_threshold);

fprintf('Call detection analysis would continue here with combined spectrogram...\n');

% Return or save results as needed
% save('detection_results.mat', 'calls', 'scores', 'F', 'T');

end

% Utility function to estimate memory usage before processing
function estimate_memory_usage(audio_file_path, chunk_duration_minutes)
% Estimate memory usage for different chunk sizes

if nargin < 2
    chunk_duration_minutes = 5;
end

% Get audio file info without loading it
info = audioinfo(audio_file_path);
total_duration_minutes = info.Duration / 60;
sample_rate = info.SampleRate;

fprintf('Audio file analysis:\n');
fprintf('- Duration: %.2f minutes\n', total_duration_minutes);
fprintf('- Sample rate: %d Hz\n', sample_rate);
fprintf('- Channels: %d\n', info.NumChannels);

% Calculate memory for different chunk sizes
chunk_sizes = [2, 5, 10, 15]; % minutes
window_size = 1024;
overlap = 512;
nfft = 1024;

fprintf('\nMemory estimates for different chunk sizes:\n');
for chunk_min = chunk_sizes
    chunk_samples = chunk_min * 60 * sample_rate;
    
    % Estimate spectrogram dimensions
    hop_length = window_size - overlap;
    n_frames = floor((chunk_samples - window_size) / hop_length) + 1;
    n_freqs = floor(nfft/2) + 1;
    
    % Memory for complex spectrogram (8 bytes per complex number)
    memory_mb = (n_frames * n_freqs * 8) / (1024^2);
    
    fprintf('  %2d min chunks: ~%.1f MB per chunk\n', chunk_min, memory_mb);
end

fprintf('\nRecommendation: Use chunks that require < 1000 MB each\n');
end