import pyaudio
import wave
import scipy.fftpack as sf
import numpy as np
import scipy.signal as scisig
import librosa
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import time



CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "output.wav"
#create "output.wav"

print("Please say something\n")
#initialize pyaudio and start recording, then save to the filename
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)
print("Recording...\n")
frames = []
for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)
print("Done recording")
time.sleep(2)
print("Saving as " + WAVE_OUTPUT_FILENAME)
stream.stop_stream()
stream.close()
p.terminate()
wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
#wf.close()

data, sampling_frequency = librosa.load("'./" + WAVE_OUTPUT_FILENAME)
#T = 1/sampling_frequency
N = len(data)
#t = N / sampling_frequency
#Y_k = np.fft.rfft(data)[0:int(N/2)]/N
#Y_k = np.fft.fft(data)[0:int(N/2)]/N
#Y_k[1:] = 2*Y_k[1:] # Single-sided spectrum
#Pxx = np.abs(Y_k) # Power spectrum

# frequencies
f = sampling_frequency * np.arange((N/2)) / N

# plotting
#fig, ax = plt.subplots()
#plt.plot(f[0:5000], Pxx[0:5000], linewidth=2)
#plt.ylabel('Amplitude')
#plt.xlabel('Frequency [Hz]')
#plt.show()

peaks = sf.find_peaks()[0] # Find peaks of the autocorrelation
lag = peaks[0] # Choose the first peak as our pitch component lag
pitch = sampling_frequency / lag  # Transform lag into frequency

print(pitch)
#save pitch to a database, then compare the pitch to all other saved entries