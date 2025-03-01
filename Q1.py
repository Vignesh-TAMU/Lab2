import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy.signal.windows as windows

# Parameters
f = 2e6  # Frequency in Hz
A = 1    # Amplitude in V
Fs_sampled = 5e6  # Sampling frequency in Hz
Fs_original = 100e6  # Higher sampling frequency for original signal
T = 3e-5  # Time duration in seconds

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
sigt_len=len(noisy_signal)
org_len=len(t_original)

# Plotting for Question - 1
plt.figure(1)

# Plot original signal
plt.plot(t_original, signal_original, label='Original Signal', color='blue', linewidth=0.5)

# Plot sampled signal as discrete points
plt.scatter(t_sampled, signal_sampled, label='Sampled Signal', color='green', s=10)

# Plot noisy sampled signal as discrete points
plt.scatter(t_sampled, noisy_signal, label='Noisy Sampled Signal', color='red', s=10)
plt.legend()
# Set plot labels and title
plt.title('Signal Simulation and Sampling with Noise')
plt.ylabel('Amplitude (V)')
plt.xlabel('Time (s)')

# Show legend


# Display the plot
#plt.show()
plt.show(block=False)

## Plotting q1a PSD graph
from numpy.fft import fft, fftfreq
X = fft(noisy_signal)
freq = fftfreq(N_sampled, 1/Fs_sampled)
psd = np.abs(X)**2 / (Fs_sampled * N_sampled)
psd_dB=10*np.log10(psd)
mask = freq >= 0
plt.figure(2)
plt.plot(freq[mask], psd[mask], label='PSD')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density (V^2/Hz)')
plt.title('Power Spectral Density of Noisy Signal')
plt.legend()
plt.show(block=False)

plt.figure(3)
plt.plot(freq[mask], psd_dB[mask], label='PSD (dB)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density (dB)')
plt.title('Power Spectral Density of Noisy Signal in dB')
plt.legend()
plt.show(block=False)

## Verification of SNR from simulation (q1a)
psdh=psd[mask] #cutting-off the negative freq. part
f_max = np.argmax(psdh)
p_net = np.sum(psdh)
p_sig = psdh[f_max]
p_noise = p_net-p_sig
SNR = 10*np.log10(p_sig/p_noise)
#print( p_net )
print("power of Peak = ", psdh[f_max] )
print( "power of Peak -1 = ", psdh[f_max-1] )
print( "power of Peak +1 = ", psdh[f_max+1] )
print( "power away from Peak = ", psdh[f_max-8] )
#print( f_max )
print(" Simulated P_signal = ", p_sig )
print(" Simulated P_noise = ", p_noise)
print(" Simulated SNR = ", SNR)
#----------END OF PART (a)-----------#

#---hann-windows---#
whann = windows.hann(sigt_len)     #######
whann_o = windows.hann(org_len)     #######
plt.figure(4)
# Plot original signal
plt.plot(t_original, signal_original*whann_o, label='Original Signal', color='blue', linewidth=0.5)  ####
# Plot noisy sampled signal as discrete points
plt.scatter(t_sampled, noisy_signal*whann, label='Noisy Sampled Signal', color='red', s=10)  ####
plt.legend()
# Set plot labels and title
plt.title('HANNING-WINDOW, Signal Simulation and Sampling with Noise')   ######
plt.ylabel('Amplitude (V)')
plt.xlabel('Time (s)')
plt.show(block=False)
## Plotting PSD graph
from numpy.fft import fft, fftfreq
X = fft(noisy_signal*whann)   #####
freq = fftfreq(N_sampled, 1/Fs_sampled)
psd = np.abs(X)**2 / (Fs_sampled * N_sampled)
psd_dB=10*np.log10(psd)
mask = freq >= 0
plt.figure(5)
plt.plot(freq[mask], psd_dB[mask], label='PSD-hanning (dB)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density (dB)')
plt.title('Power Spectral Density of Hanning windowed Noisy Signal in dB')  ####
plt.legend()
plt.show(block=False)
## SNR Verification
psdh=psd[mask] #cutting-off the negative freq. part
f_max = np.argmax(psdh)
p_net = np.sum(psdh)
p_sig = psdh[f_max] + psdh[f_max-1] + psdh[f_max+1] + psdh[f_max-2] + psdh[f_max+2] 
p_noise = p_net-p_sig
SNR = 10*np.log10(p_sig/p_noise)
print("---HANNING WINDOW---")
print("power of Peak = ", psdh[f_max] )
print( "power of Peak -1 = ", psdh[f_max-1] )
print( "power of Peak +1 = ", psdh[f_max+1] )
print( "power of Peak -2 = ", psdh[f_max-2] )
print( "power of Peak +2 = ", psdh[f_max+2] )
print( "power of Peak -3 = ", psdh[f_max-3] )
print( "power of Peak +3 = ", psdh[f_max+3] )
print( "power away from Peak (8th) = ", psdh[f_max-8] )
#print( f_max )
print(" Simulated P_signal = ", p_sig )
print(" Simulated P_noise = ", p_noise)
print(" Simulated SNR = ", SNR)

#---hamming-windows---#
whamm = windows.hamming(sigt_len)     #######
whamm_o = windows.hamming(org_len)     #######
plt.figure(6)
# Plot original signal
plt.plot(t_original, signal_original*whamm_o, label='Original Signal', color='blue', linewidth=0.5)  ####
# Plot noisy sampled signal as discrete points
plt.scatter(t_sampled, noisy_signal*whamm, label='Noisy Sampled Signal', color='red', s=10)  ####
plt.legend()
# Set plot labels and title
plt.title('HAMMING-WINDOW, Signal Simulation and Sampling with Noise')   ######
plt.ylabel('Amplitude (V)')
plt.xlabel('Time (s)')
plt.show(block=False)
## Plotting PSD graph
from numpy.fft import fft, fftfreq
X = fft(noisy_signal*whamm)   #####
freq = fftfreq(N_sampled, 1/Fs_sampled)
psd = np.abs(X)**2 / (Fs_sampled * N_sampled)
psd_dB=10*np.log10(psd)
mask = freq >= 0
plt.figure(7)
plt.plot(freq[mask], psd_dB[mask], label='PSD-hamming (dB)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density (dB)')
plt.title('Power Spectral Density of Hamming windowed Noisy Signal in dB')  ####
plt.legend()
plt.show(block=False)
## SNR Verification
psdh=psd[mask] #cutting-off the negative freq. part
f_max = np.argmax(psdh)
p_net = np.sum(psdh)
p_sig = psdh[f_max] + psdh[f_max-1] + psdh[f_max+1] + psdh[f_max-2] + psdh[f_max+2] 
p_noise = p_net-p_sig
SNR = 10*np.log10(p_sig/p_noise)
print("---HAMMING WINDOW---") ####
print("power of Peak = ", psdh[f_max] )
print( "power of Peak -1 = ", psdh[f_max-1] )
print( "power of Peak +1 = ", psdh[f_max+1] )
print( "power of Peak -2 = ", psdh[f_max-2] )
print( "power of Peak +2 = ", psdh[f_max+2] )
print( "power of Peak -3 = ", psdh[f_max-3] )
print( "power of Peak +3 = ", psdh[f_max+3] )
print( "power away from Peak (8th) = ", psdh[f_max-8] )
#print( f_max )
print(" Simulated P_signal = ", p_sig )
print(" Simulated P_noise = ", p_noise)
print(" Simulated SNR = ", SNR)

#---Blackman Window---#
wblkm = windows.blackman(sigt_len)     #######
wblkm_o = windows.blackman(org_len)     #######
plt.figure(8)
# Plot original signal
plt.plot(t_original, signal_original*wblkm_o, label='Original Signal', color='blue', linewidth=0.5)  ####
# Plot noisy sampled signal as discrete points
plt.scatter(t_sampled, noisy_signal*wblkm, label='Noisy Sampled Signal', color='red', s=10)  ####
plt.legend()
# Set plot labels and title
plt.title('BLACKMAN-WINDOW, Signal Simulation and Sampling with Noise')   ######
plt.ylabel('Amplitude (V)')
plt.xlabel('Time (s)')
plt.show(block=False)
## Plotting PSD graph
from numpy.fft import fft, fftfreq
X = fft(noisy_signal*wblkm)   #####
freq = fftfreq(N_sampled, 1/Fs_sampled)
psd = np.abs(X)**2 / (Fs_sampled * N_sampled)
psd_dB=10*np.log10(psd)
mask = freq >= 0
plt.figure(9)
plt.plot(freq[mask], psd_dB[mask], label='PSD-blackman (dB)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density (dB)')
plt.title('Power Spectral Density of Blackman windowed Noisy Signal in dB')  ####
plt.legend()
plt.show(block=False)
## SNR Verification
psdh=psd[mask] #cutting-off the negative freq. part
f_max = np.argmax(psdh)
p_net = np.sum(psdh)
p_sig = psdh[f_max] + psdh[f_max-1] + psdh[f_max+1] + psdh[f_max-2] + psdh[f_max+2] 
p_noise = p_net-p_sig
SNR = 10*np.log10(p_sig/p_noise)
print("---BLACKMAN WINDOW---")  ####
print("power of Peak = ", psdh[f_max] )
print( "power of Peak -1 = ", psdh[f_max-1] )
print( "power of Peak +1 = ", psdh[f_max+1] )
print( "power of Peak -2 = ", psdh[f_max-2] )
print( "power of Peak +2 = ", psdh[f_max+2] )
print( "power of Peak -3 = ", psdh[f_max-3] )
print( "power of Peak +3 = ", psdh[f_max+3] )
print( "power away from Peak (8th) = ", psdh[f_max-8] )
#print( f_max )
print(" Simulated P_signal = ", p_sig )
print(" Simulated P_noise = ", p_noise)
print(" Simulated SNR = ", SNR)
print("--------- END OF SIM ---------")
plt.show()
