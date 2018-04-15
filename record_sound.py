import numpy as np
from tempfile import TemporaryFile
import  sounddevice as sd
import scipy.io.wavfile as sciwav

if __name__ =="__main__":
    duration=4
    rate=44100
    print ("start recording")
    record=sd.rec(int(duration*rate),samplerate=rate,channels=2)
    sd.wait()
    print ("end recording")
    print (record)
    sd.play(record, rate)
    sciwav.write("recorded.wav", rate, record)