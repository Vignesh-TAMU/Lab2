#print("Helo world")
import numpy as np
import matplotlib.pyplot as plt

f = 2e6  # Frequency in Hz (2 MHz)
A = 1    # Amplitude in V
Fs = 5e6 # Sampling frequency in Hz (5 MHz)
T = 1    # Time duration in seconds

# Number of samples
N = int(Fs * T)

# Time array
t = np.linspace(0, T, N, endpoint=False)

# Generate the signal (sampled at Fs)
signal = A * np.sin(2 * np.pi * f * t)

# Optionally, plot a small portion for visualization (e.g., first 0.00001 seconds)
plt.figure(figsize=(10, 4))
plt.plot(t[:250], signal[:250],'o', markersize=1, linestyle='none')  # Plot first 250 points (0.00005 seconds)
plt.title("First Few Cycles of 2 MHz Tone (Sampled at 5 MHz)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (V)")
plt.grid(True)
plt.show()