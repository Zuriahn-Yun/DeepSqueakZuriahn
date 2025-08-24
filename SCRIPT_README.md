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

## Support and Contributions

### Getting Help
- Check the DeepSqueak documentation for function details
- Review MATLAB error messages for specific issues
- Ensure all dependencies are properly installed

### Contributing
- Test with different audio types and network architectures
- Report bugs and suggest improvements
- Share custom network integration examples

## License

These scripts are part of the DeepSqueak project and follow the same licensing terms.

## Acknowledgments

- DeepSqueak development team for the core functionality
- MATLAB community for neural network and clustering tools
- Contributors to the DeepSqueak ecosystem
