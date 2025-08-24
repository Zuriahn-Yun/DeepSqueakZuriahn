# DeepSqueak Scripts Usage Guide

## Quick Start

### 1. Single File Processing with Custom Network
Your `script.m` is now configured to use your `NNTest#1.mat` network for clustering!

**To run:**
```matlab
% Just run the script
run('script.m')
```

**What it does:**
- Loads your audio file (`Audio/VL1_25-07-19.wav`)
- Detects calls using the mouse detection network
- Creates spectrograms around each detected call
- Uses your trained VAE network (`NNTest#1.mat`) for clustering
- Saves results to the `output/` folder

### 2. Batch Processing Multiple Folders
**To process multiple audio folders:**

1. **Edit the configuration:**
   ```matlab
   % Open batch_config.m and modify:
   base_directory = 'Audio/';  % Your audio base directory
   output_base_directory = 'batch_output/';  % Where to save results
   ```

2. **Run batch processing:**
   ```matlab
   run('batch_process_folders.m')
   ```

**What it does:**
- Scans your `Audio/` folder for subfolders
- Processes each folder sequentially (one after another)
- Uses your custom VAE network for clustering
- Saves results in organized subfolders

## Your Custom Network Integration

### Network Structure
Your `NNTest#1.mat` contains:
- `encoderNet` - VAE encoder for feature extraction
- `decoderNet` - VAE decoder (not used for clustering)
- `options` - Network configuration (image size, frequency range, etc.)
- `C` - Pre-trained cluster centers (4 clusters)
- `clusterName` - Names/labels for each cluster

### How Clustering Works
1. **Feature Extraction**: Each call spectrogram is processed through your VAE encoder
2. **Clustering**: The extracted features are compared to your pre-trained cluster centers
3. **Assignment**: Each call is assigned to the nearest cluster center
4. **Results**: You get cluster assignments and distances for each call

## Configuration Options

### Detection Settings
```matlab
detection_settings = {
    '0',      % 0 = full file, or specify duration in seconds
    '100',    % High frequency cutoff (kHz)
    '18',     % Low frequency cutoff (kHz)
    '0.5',   % Score threshold (0.5 = moderate confidence)
    '1'       % Enable detection
};
```

### Network Selection
```matlab
% Choose your detection network:
network_file = 'Networks/2025.02.18.Mouse.YoloR3.mat';  % Mouse
% network_file = 'Networks/2025.02.04.Rat.YoloR2.mat';  % Rat
% network_file = 'Networks/2025.02.05.Rat.Long.YoloR2.mat';  % Long rat
```

### Clustering Method
```matlab
clustering_method = 'VAE';  % Use your custom network
% clustering_method = 'contour';  % Use simple feature-based clustering
```

## Output Structure

### Single File Processing
```
output/
├── detected_calls.mat          # Raw detection results
├── clustering_data.mat         # Processed spectrograms
├── final_results.mat           # Complete results with clustering
└── processing_report.txt       # Summary report
```

### Batch Processing
```
batch_output/
├── folder1/
│   ├── file1_detected_calls.mat
│   ├── file1_clustering_data.mat
│   ├── file1_final_results.mat
│   └── file1_report.txt
├── folder2/
│   └── ...
└── base_directory/
    └── ...
```

## Troubleshooting

### Common Issues

1. **"Network file not found"**
   - Check that `NNTest#1.mat` is in your main directory
   - Verify the path in the script

2. **"VAE clustering failed"**
   - The script will automatically fall back to contour-based clustering
   - Check that your VAE network is compatible with the input format

3. **"No calls detected"**
   - Try lowering the score threshold (e.g., from 0.5 to 0.3)
   - Check that your audio file contains the expected vocalizations
   - Verify the frequency range settings match your audio

4. **Memory issues with large files**
   - Set `max_audio_duration` in `batch_config.m` to limit processing time
   - Process files individually instead of in batch

### Performance Tips

1. **For large datasets**: Use batch processing with `skip_existing_output = true`
2. **For faster processing**: Enable parallel processing if you have the Parallel Computing Toolbox
3. **For memory management**: Process shorter audio segments by setting `detection_settings{1}` to a specific duration

## Customization

### Adding New Audio Formats
Edit `batch_config.m` and modify the `audio_extensions` array:
```matlab
audio_extensions = {
    '*.wav',   % WAV files
    '*.flac',  % FLAC files
    '*.mp3',   % MP3 files
    '*.m4a',   % M4A files
    '*.aiff',  % AIFF files
    '*.ogg',   % OGG files
    '*.your_format'  % Add your format here
};
```

### Modifying Clustering Parameters
Your VAE network already has optimized parameters, but you can adjust:
- **Image size**: Modify `options.imageSize` in your network file
- **Frequency range**: Adjust `options.freqRange` 
- **Duration scaling**: Modify `options.maxDuration`

### Adding New Detection Networks
1. Place your network file in the `Networks/` folder
2. Update `network_file` in `batch_config.m`
3. Ensure compatibility with the `SqueakDetect` function

## Example Workflows

### Workflow 1: Single File Analysis
```matlab
% Edit script.m configuration section
audio_file = 'path/to/your/audio.wav';
network_file = 'Networks/2025.02.18.Mouse.YoloR3.mat';
custom_network_path = 'NNTest#1.mat';

% Run the script
run('script.m')
```

### Workflow 2: Batch Analysis
```matlab
% Edit batch_config.m
base_directory = 'Audio/';
clustering_method = 'VAE';
custom_network_path = 'NNTest#1.mat';

% Run batch processing
run('batch_process_folders.m')
```

### Workflow 3: Resume Interrupted Processing
```matlab
% Edit batch_config.m
skip_existing_output = true;

% Run batch processing (will skip already processed files)
run('batch_process_folders.m')
```

## Support

If you encounter issues:
1. Check the error messages in the MATLAB console
2. Verify all file paths are correct
3. Ensure your VAE network is compatible with the input format
4. Check that all required MATLAB toolboxes are installed

Your custom network should work seamlessly with these scripts for both single file and batch processing!
