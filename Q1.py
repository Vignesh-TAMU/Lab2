import numpy as np
import matplotlib.pyplot as plt

# Parameters
f = 2e6  # Frequency in Hz
A = 1    # Amplitude in V
Fs_sampled = 5e6  # Sampling frequency in Hz
Fs_original = 100e6  # Higher sampling frequency for original signal
T = 1e-5  # Time duration in seconds

# Number of samples
N_sampled = int(Fs_sampled * T)
N_original = int(Fs_original * T)

# Time arrays
t_original = np.linspace(0, T, N_original, endpoint=False)
t_sampled = np.linspace(0, T, N_sampled, endpoint=False)

# Generate original signal
signal_original = A * np.sin(2 * np.pi * f * t_original)

# Generate sampled signal
k = int(Fs_original / Fs_sampled)
signal_sampled = signal_original[0::k]

# Calculate the power of the signal
P_signal = np.mean(signal_sampled**2)

# Desired SNR in dB
SNR_dB = 50
SNR = 10**(SNR_dB / 10)
P_Noise = P_signal / SNR
sigma = np.sqrt(P_Noise)

# Generate Gaussian noise
np.random.seed(42)  # For reproducibility
noise = np.random.normal(0, sigma, size=N_sampled)

# Generate noisy sampled signal
noisy_signal = signal_sampled + noise

# Plotting
plt.figure()

# Plot original signal
plt.plot(t_original, signal_original, label='Original Signal', color='blue', linewidth=0.5)

# Plot sampled signal as discrete points
plt.scatter(t_sampled, signal_sampled, label='Sampled Signal', color='green', s=10)

# Plot noisy sampled signal as discrete points
plt.scatter(t_sampled, noisy_signal, label='Noisy Sampled Signal', color='red', s=10)

# Set plot labels and title
plt.title('Signal Simulation and Sampling with Noise')
plt.ylabel('Amplitude (V)')
plt.xlabel('Time (s)')

# Show legend
plt.legend()

# Display the plot
plt.show()
