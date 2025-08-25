# DeepSqueak Headless Processing Suite

A comprehensive MATLAB suite for processing audio files using DeepSqueak's core functions without requiring the GUI interface. This suite provides both single-file and batch processing capabilities with advanced VAE-based clustering.

# Line 9 -> Pick audio File 
# For Batch Processing edit the batch_config file with the audio folder

## Features

- **Headless Processing**: Run DeepSqueak analysis without GUI
- **Neural Network Detection**: Uses Faster-RCNN networks for call detection
- **VAE-Based Clustering**: Advanced clustering using Variational Autoencoders
- **Batch Processing**: Process multiple audio files and folders sequentially
- **CSV Export**: Export cluster results in CSV format for analysis
- **DeepSqueak Compatibility**: Uses exact same preprocessing and clustering as DeepSqueak GUI

## File Structure

```
DeepSqueakZuriahn/
├── script.m                          # Single audio file processing
├── batch_process_folders.m           # Batch processing for multiple folders
├── batch_config.m                    # Configuration file for batch processing
├── analyze_clusters.m                # Analysis and visualization of results
├── USAGE_GUIDE.md                    # Detailed usage instructions
├── Audio/                            # Place your audio files here
├── Networks/                         # Detection networks (.mat files)
├── output/                           # Results from single file processing
└── batch_output/                     # Results from batch processing
```

##  Quick Start

### Single File Processing
```matlab
% Run the main script
run('script.m')

% Or modify script.m to change audio file path
audio_file = 'Audio/your_file.wav';
```

### Batch Processing
```matlab
% Run batch processing
run('batch_process_folders.m')

% Or modify batch_config.m for custom settings
```

## Configuration

### Single File Settings (script.m)
- **Audio File**: Path to your audio file
- **Network**: Detection network (.mat file)
- **Detection Settings**: Frequency cutoffs, score thresholds
- **Clustering Method**: VAE or contour-based

### Batch Processing Settings (batch_config.m)
- **Base Directory**: Root folder containing audio files
- **Output Directory**: Where to save results
- **Network File**: Detection network to use
- **Custom Network**: VAE clustering network path
- **Audio Extensions**: Supported file formats

## VAE Clustering

The VAE clustering system works exactly like DeepSqueak's GUI:

1. **Feature Extraction**: 32-dimensional VAE embeddings
2. **Frequency Contours**: 16-dimensional frequency features
3. **Feature Combination**: 48-dimensional total features
4. **Z-Score Normalization**: Applied to both feature types
5. **Clustering**: Uses pre-trained cluster centers with k-nearest neighbors

### Network Requirements
Your VAE network must contain:
- `encoderNet`: Trained encoder network
- `options`: Configuration (imageSize, freqRange, maxDuration)
- `C`: Pre-trained cluster centers (4×48 matrix)
- `clusterName`: Names for each cluster

##  Output

### Single File Processing
- `output/final_results.mat`: Complete results
- `output/cluster_results.csv`: Cluster counts in CSV format
- `output/processing_report.txt`: Detailed processing log

### Batch Processing
- `batch_output/folder_name/filename_final_results.mat`: Per-file results
- `batch_output/folder_name/filename_cluster_results.csv`: Per-file CSV exports
- `batch_output/folder_name/filename_report.txt`: Per-file processing logs

### CSV Format
```csv
AudioFile,Cluster1,Cluster2,Cluster3,Cluster4,TotalCalls,FailedCalls
VL1_25-07-19,15,12,8,21,56,0
```

## 🛠️ Requirements

- **MATLAB**: R2019b or later
- **Deep Learning Toolbox**: For neural network operations
- **Signal Processing Toolbox**: For audio processing
- **Image Processing Toolbox**: For spectrogram operations
- **Statistics and Machine Learning Toolbox**: For clustering

## Usage Examples

### Example 1: Process Single Audio File
```matlab
% Edit script.m to set your audio file
audio_file = 'Audio/my_recording.wav';
network_file = 'Networks/my_detector.mat';

% Run processing
run('script.m');
```

### Example 2: Batch Process Multiple Folders
```matlab
% Edit batch_config.m to set directories
base_directory = 'Audio/';
output_base_directory = 'batch_output/';
custom_network_path = 'my_vae_network.mat';

% Run batch processing
run('batch_process_folders.m');
```

### Example 3: Analyze Results
```matlab
% Load and analyze saved results
run('analyze_clusters.m');
```


### Debug Mode
Enable verbose output by running scripts directly in MATLAB command window to see detailed progress information.

## Documentation

- **USAGE_GUIDE.md**: Comprehensive usage instructions
- **SCRIPT_README.md**: Technical details about script implementation
- **analyze_clusters.m**: Built-in help for analysis functions

**Note**: This suite is designed to replicate DeepSqueak's GUI functionality in a headless environment. All preprocessing, feature extraction, and clustering methods are identical to the GUI version.


SCRIPT_README----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# DeepSqueak Headless Processing Scripts

This repository contains MATLAB scripts that allow you to use DeepSqueak's core functionality without the GUI interface. These scripts are designed for batch processing, automation, and integration with custom neural networks.

## Overview



The scripts provide a complete pipeline for:
1. **Audio Processing**: Loading and analyzing audio files
2. **Call Detection**: Using pre-trained neural networks to detect vocalizations
3. **Spectrogram Generation**: Creating focused spectrograms around detected calls
4. **Feature Extraction**: Extracting call images and metadata
5. **Clustering**: Grouping similar calls using various methods
6. **Custom Network Integration**: Using your own trained neural networks for clustering

## Files

### Main Script
- **`script.m`** - Complete pipeline script that processes audio files end-to-end

### Helper Scripts
- **`custom_clustering_integration.m`** - Function for integrating custom neural networks
- **`SCRIPT_README.md`** - This documentation file

## Prerequisites

### Required MATLAB Toolboxes
- **Deep Learning Toolbox** - For neural network operations
- **Signal Processing Toolbox** - For audio and spectrogram processing
- **Image Processing Toolbox** - For image manipulation and clustering
- **Statistics and Machine Learning Toolbox** - For clustering algorithms

### DeepSqueak Dependencies
Ensure you have the complete DeepSqueak codebase in your MATLAB path, including:
- `Functions/` directory with all DeepSqueak functions
- `Networks/` directory with pre-trained detection networks
- Required helper functions and classes

## Quick Start

### 1. Basic Usage

```matlab
% Edit the configuration section in script.m
audio_file = 'path/to/your/audio.wav';
network_file = 'Networks/2025.02.18.Mouse.YoloR3.mat';
output_folder = 'output/';

% Run the script
run('script.m');
```

### 2. Custom Network Integration

```matlab
% In script.m, modify the custom network path
custom_network_path = 'path/to/your/trained/network.mat';

% The script will automatically attempt to use your network for clustering
```

## Configuration Options

### Audio Settings
- **Detection Length**: Set to `'0'` for full file analysis, or specify duration in seconds
- **Frequency Range**: Set high and low frequency cutoffs in kHz
- **Score Threshold**: Minimum confidence score for call detection
- **Enable Detection**: Toggle detection on/off

### Clustering Settings
- **Method**: Choose between `'VAE'` (Variational Autoencoder) or `'contour'` (feature-based)
- **Output**: Enable/disable result saving and reporting

### Network Selection
Choose from available pre-trained networks:
- `2025.02.18.Mouse.YoloR3.mat` - Mouse vocalization detector
- `2025.02.05.Rat.Long.YoloR2.mat` - Long rat vocalization detector
- `2025.02.04.Rat.YoloR2.mat` - Rat vocalization detector

## Custom Neural Network Integration

### Supported Network Formats
The scripts support various neural network formats:

1. **MATLAB Network Objects** (`network`)
2. **Deep Learning Toolbox Networks** (`dlnetwork`, `SeriesNetwork`)
3. **Custom Architectures** (VAE, Autoencoders, etc.)

### Integration Steps

1. **Prepare Your Network**:
   - Train your neural network for call clustering
   - Save it as a `.mat` file
   - Ensure it can process spectrogram inputs

2. **Modify the Script**:
   - Update `custom_network_path` in `script.m`
   - Adjust preprocessing in `custom_clustering_integration.m`
   - Modify feature extraction based on your network architecture

3. **Test Integration**:
   - Run with a small audio file first
   - Verify feature extraction works correctly
   - Check clustering results

### Example Custom Network Structure

```matlab
% Your network should be saved with a structure like:
network_data = struct();
network_data.net = your_trained_network;  % The actual network
network_data.input_size = [64, 64];       % Expected input dimensions
network_data.feature_size = 128;          % Output feature dimensions

% Or for VAE/autoencoder:
network_data.encoder = encoder_network;
network_data.decoder = decoder_network;
network_data.options = training_options;
```

## Output Structure

### Generated Files
- **`detected_calls.mat`** - Raw detection results
- **`clustering_data.mat`** - Processed spectrograms and metadata
- **`final_results.mat`** - Complete results including clustering
- **`processing_report.txt`** - Summary of processing steps

### Data Structures

#### Calls Table
Contains detected call information:
- `Box`: [start_time, start_freq, duration, bandwidth]
- `Score`: Detection confidence
- `Accept`: Call validation status
- Additional metadata fields

#### ClusteringData Table
Contains processed call data:
- `Spectrogram`: Cell array of call spectrograms
- `Box`: Call bounding boxes
- `Cluster`: Cluster assignments (after clustering)
- `xFreq`, `xTime`: Frequency and time contours (if available)

## Advanced Usage

### Batch Processing
```matlab
% Process multiple audio files
audio_files = {'file1.wav', 'file2.wav', 'file3.wav'};
for i = 1:length(audio_files)
    audio_file = audio_files{i};
    output_folder = sprintf('output/file_%d/', i);
    run('script.m');
end
```

### Custom Preprocessing
Modify the `preprocess_spectrogram_for_network` function in `custom_clustering_integration.m` to match your network's input requirements.

### Alternative Clustering Methods
The scripts support multiple clustering approaches:
- **K-means**: Default method with automatic cluster number estimation
- **Hierarchical**: Uncomment in the clustering section
- **DBSCAN**: Density-based clustering for irregular cluster shapes

## Troubleshooting

### Common Issues

1. **"Network file not found"**
   - Check the network file path in the configuration
   - Ensure the Networks/ directory is accessible

2. **"Audio file not found"**
   - Verify the audio file path is correct
   - Check file permissions and format support

3. **"No calls detected"**
   - Verify the neural network is appropriate for your audio
   - Check detection settings (frequency range, score threshold)
   - Ensure audio quality meets detection requirements

4. **"VAE clustering failed"**
   - The script will automatically fall back to contour-based clustering
   - For VAE clustering, ensure all required functions are available

5. **"Custom network integration failed"**
   - Check network file format and structure
   - Verify preprocessing matches network input requirements
   - Ensure all required toolboxes are installed

### Performance Optimization

- **Memory Management**: For large audio files, consider processing in chunks
- **Parallel Processing**: Use `parfor` loops for spectrogram generation
- **GPU Acceleration**: Ensure Deep Learning Toolbox GPU support is enabled

## Extending the Scripts

### Adding New Detection Networks
1. Place your network file in the `Networks/` directory
2. Update the network selection in the configuration
3. Ensure compatibility with the `SqueakDetect` function

### Adding New Clustering Methods
1. Implement your clustering algorithm
2. Add it to the clustering switch statement in `script.m`
3. Ensure output format matches existing clustering methods

### Custom Feature Extraction
1. Modify the `extract_features_with_network` function
2. Implement your feature extraction logic
3. Ensure output dimensions are consistent


USAGE_GUIDE -------------------------------------------------------------------------------------------------------------------------------------------------------------------

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
