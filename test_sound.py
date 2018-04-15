import scipy.io.wavfile as sciwav
import sounddevice as sd
import sys

if __name__ == "__main__":
    rate, data = sciwav.read(sys.argv[1])
    sd.play(data, rate, blocking=True)
