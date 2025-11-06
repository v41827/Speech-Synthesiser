import soundfile as sf
import numpy as np

import matplotlib.pyplot as plt

# Read the audio file
# Replace 'audio_file.wav' with your audio file path
data, fs = sf.read('/Users/yiwenchan/Desktop/Workspace/speech_segments_100ms/heed_f_segment.wav')

# Create time array
time = np.arange(len(data))/fs #in seconds

# Create the plot
plt.figure(figsize=(12, 4))
plt.plot(time, data)
plt.title('Audio Waveform')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()