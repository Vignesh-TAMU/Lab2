import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import scipy.signal.windows as wnd

# System parameters
FREQ_SAMPLING = 640  # Measured in MHz
PERIOD = 1 / (FREQ_SAMPLING * 1e6)
DURATION = 30
RESOLUTION = 12
BINS_FOR_SIGNAL = 100

# Time base creation
time_steps = len(np.arange(0, DURATION / 200e6, PERIOD / 100))
continuous_time = np.linspace(0, DURATION / 200e6, time_steps)
base_wave = np.cos(2 * np.pi * 200e6 * continuous_time)

discrete_points = np.arange(0, DURATION / 200e6 - PERIOD, PERIOD)
sampled_wave = np.cos(2 * np.pi * 200e6 * discrete_points)

sample_count = len(sampled_wave)

# Noise generation
center = 0
spread = np.sqrt(7.9244e-5)
interference = np.random.normal(center, spread, sampled_wave.shape)
disturbed_signal = sampled_wave + interference

# Window application
window_shape = wnd.hann(sample_count) - np.mean(wnd.hann(sample_count))
filtered_data = window_shape * disturbed_signal
quantization_level = 1 / (2 ** RESOLUTION)
print(quantization_level)
quantized_result = np.round(filtered_data / quantization_level) * quantization_level

# Spectral analysis
spectrum, frequencies = plt.psd(quantized_result,
                               NFFT=sample_count,
                               Fs=1/PERIOD,
                               color='red',
                               sides='default',
                               pad_to=2*sample_count)
plt.title('Frequency Power Analysis')

peak_index = np.argmax(spectrum)
print(peak_index)
total_energy = np.sum(spectrum)
signal_energy = np.sum(spectrum[peak_index-BINS_FOR_SIGNAL:peak_index+BINS_FOR_SIGNAL])
background_noise = total_energy - signal_energy
signal_to_noise = 10 * np.log10(signal_energy / background_noise)

print(f"Signal Level: {10*np.log10(signal_energy):.3f} dB")
print(f"Noise Level: {10*np.log10(background_noise):.3f} dB")
print(f"SNR from Spectrum: {signal_to_noise:.3f} dB")

# Visualization setup
canvas, (graph1, graph2, graph3) = plt.subplots(3, 1, constrained_layout=True)

graph1.plot(continuous_time, base_wave, color='red')
graph1.set_title('Pure 200MHz Oscillation')
graph1.set_xlabel('Seconds')
graph1.set_ylabel('Voltage')
graph1.grid(True)

graph2.plot(discrete_points, sampled_wave, color='blue')
graph2.set_title(f'Sampled Waveform at {FREQ_SAMPLING} MHz')
graph2.set_xlabel('Seconds')
graph2.set_ylabel('Voltage')
graph2.grid(True)

graph3.plot(discrete_points, quantized_result, color='green')
graph3.set_title('Quantized Output with Noise and Windowing')
graph3.set_xlabel('Seconds')
graph3.set_ylabel('Voltage')
graph3.grid(True)

plt.show()
