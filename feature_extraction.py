import audioFeatureExtraction
import audioBasicIO
import numpy as np
import matplotlib.pyplot as plt

def tutorial():
    [Fs, x] = audioBasicIO.readAudioFile("/home/parallels/git_repos/pyAudioAnalysis/data/scottish.wav")
    F = audioFeatureExtraction.stFeatureExtraction(x, Fs, 0.050 * Fs, 0.025 * Fs)
    plt.subplot(2, 1, 1)
    plt.plot(F[0, :])
    plt.xlabel('Frame no')
    plt.ylabel('ZCR')
    plt.subplot(2, 1, 2)
    plt.plot(F[1, :])
    plt.xlabel('Frame no')
    plt.ylabel('Energy')
    plt.savefig('plot1.png', format='png')



if __name__ == "__main__":
    np.load("/home/parallels/git_repos/pyAudioAnalysis/data/speech_music_sample.wav.npy")
    p