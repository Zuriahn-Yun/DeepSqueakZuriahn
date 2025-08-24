# DeepSqueak Headless Processing Suite

A comprehensive MATLAB suite for processing audio files using DeepSqueak's core functions without requiring the GUI interface. This suite provides both single-file and batch processing capabilities with advanced VAE-based clustering.

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