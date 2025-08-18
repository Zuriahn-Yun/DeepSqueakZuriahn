Zuriahn Fork

Goal?
Set up a supervised classiefier for mouse USVs. 


Training works but issues with the actual classifier. 

Triying to do USV and call classification. 

User error 

I tried the unsupervised version, but for both supervised and unsupervised clustering, I get to the point where the call clusters come up and I can accept/reject/rename them, but after that, running other audio/detection files through the model doesn't work. 

Technical Error

Variable wind is not found..... 
Steps? 



- Create a pipeline to do mass analysis, start with creating the images and be able to improt a trained nn to cluster them.

Recommended Exploration Order:

Start with FILE 47 (CreateClusteringData.m) - this orchestrates the full pipeline
Examine FILE 13 and FILE 14 (the spectrogram creation functions)
Look at FILE 42 (squeakData.m) to understand the data structures
Check FILE 52 to see how spectrograms feed into classification

