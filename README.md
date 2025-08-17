Zuriahn Fork

- Create a pipeline to do mass analysis, start with creating the images and be able to improt a trained nn to cluster them.

Recommended Exploration Order:

Start with FILE 47 (CreateClusteringData.m) - this orchestrates the full pipeline
Examine FILE 13 and FILE 14 (the spectrogram creation functions)
Look at FILE 42 (squeakData.m) to understand the data structures
Check FILE 52 to see how spectrograms feed into classification