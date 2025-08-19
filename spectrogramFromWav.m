function spectrogramFromWav(filename)
    [audio, rate] = audioread(filename);
    

    windowsize = 1024;
    noverlap   = 512;
    nfft       = 1024;

    [S,F,T,P] = spectrogram(audio, windowsize, noverlap, nfft, rate, 'yaxis');

    imagesc(T, F, 10*log10(abs(P)));
    axis xy;
    xlabel('Time (s)');
    ylabel('Frequency (Hz)');
    colorbar;
    title(['Spectrogram of ', filename]);

    % Call on local machine -> spectrogramFromWav("Audio/VL1_25-07-19.wav")
    % Displays a spectrogram....
end
