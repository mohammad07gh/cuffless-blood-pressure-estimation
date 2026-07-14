import wfdb
import os 

folder_path = "D:/New folder (6)" 


base_file_name = "s10_run" 

full_path = os.path.join(folder_path, base_file_name)

print(f"try to read: {full_path}")

try:
    record = wfdb.rdrecord(full_path) 
    signals = record.p_signal
    fs = record.fs
    signal_names = record.sig_name
    print("\n read successfully")
    print(f"(fs): {fs} Hz")
    print(f"signal names: {signal_names}")

except Exception as e:
    print(f"\n error")
    print(f"why: {e}")
    print("\n do ")
    print("1. address")
    print("2. name")
    print("3. data")
    
import matplotlib.pyplot as plt

try:
    ecg_index = record.sig_name.index('ecg')
    ppg_index = record.sig_name.index('pleth_1')
    ecg_signal = record.p_signal[:, ecg_index]
    ppg_signal = record.p_signal[:, ppg_index]

    fig, axs = plt.subplots(2, 1, figsize=(17, 8), sharex=True)

    axs[0].plot(ecg_signal, label='ECG')
    axs[0].set_title('ECG Signal')
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(ppg_signal, label='PPG (pleth_1)', color='red')
    axs[1].set_title('PPG Signal')
    axs[1].legend()
    axs[1].grid(True)

    plt.xlabel('Sample Number')
    plt.tight_layout()
    plt.show()

except ValueError as e:
    print(f"one of signals does not find: {e}")
    
from scipy.signal import butter, filtfilt
import numpy as np
import matplotlib.pyplot as plt

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y



fs = record.fs 
ecg_signal = record.p_signal[:, record.sig_name.index('ecg')]
ppg_signal = record.p_signal[:, record.sig_name.index('pleth_1')]
ecg_filtered = bandpass_filter(ecg_signal, lowcut=0.5, highcut=40.0, fs=fs)
ppg_filtered = bandpass_filter(ppg_signal, lowcut=0.5, highcut=4.0, fs=fs)



num_samples_to_plot = 3 * fs 

fig, axs = plt.subplots(2, 1, figsize=(17, 8), sharex=True)
axs[0].plot(ecg_signal[:num_samples_to_plot], label='ECG before filter', color='silver', alpha=0.7)
axs[0].plot(ecg_filtered[:num_samples_to_plot], label='ECG Filtered', color='blue')
axs[0].set_title('comparison of ECG')
axs[0].legend()
axs[0].grid(True)

axs[1].plot(ppg_signal[:num_samples_to_plot], label='PPG Raw', color='silver', alpha=0.7)
axs[1].plot(ppg_filtered[:num_samples_to_plot], label='PPG Filtered', color='red')
axs[1].set_title('comparison PPG')
axs[1].legend()
axs[1].grid(True)

plt.xlabel('Sample Number')
plt.tight_layout()
plt.show()

from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import numpy as np

peaks, _ = find_peaks(ecg_filtered, height=np.mean(ecg_filtered) + 0.5 * np.std(ecg_filtered), distance=150)

num_samples_to_plot = 3 * fs

peaks_in_plot = peaks[peaks < num_samples_to_plot]

plt.figure(figsize=(17, 5))
plt.plot(ecg_filtered[:num_samples_to_plot], label='ECG Filtered')
plt.plot(peaks_in_plot, ecg_filtered[peaks_in_plot], "x", color='red', markersize=10, label='R')
plt.title('Finding R in ECG')
plt.xlabel('Sample Number')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.show()

print(f"\n {len(peaks)} Number of R found")

ppg_peaks, _ = find_peaks(ppg_filtered, height=np.mean(ppg_filtered), distance=150)
ppg_troughs, _ = find_peaks(-ppg_filtered, height=-np.mean(ppg_filtered), distance=150)
num_samples_to_plot = 3 * fs
ppg_peaks_in_plot = ppg_peaks[ppg_peaks < num_samples_to_plot]
ppg_troughs_in_plot = ppg_troughs[ppg_troughs < num_samples_to_plot]

plt.figure(figsize=(17, 5))
plt.plot(ppg_filtered[:num_samples_to_plot], label='PPG Filtered')
plt.plot(ppg_peaks_in_plot, ppg_filtered[ppg_peaks_in_plot], "x", color='blue', markersize=10, label='Peaks of PPG')
plt.plot(ppg_troughs_in_plot, ppg_filtered[ppg_troughs_in_plot], "o", color='red', markersize=10, label='PPG Troughs')
plt.title('Finding peaks of PPG')
plt.xlabel('Sample Number')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.show()

print(f"\n {len(ppg_troughs)} ")

ptt_values_ms = [] 
ptt_peak_times = [] 

for r_peak in peaks:
   
    future_troughs_indices = np.where(ppg_troughs > r_peak)[0]
    
    if len(future_troughs_indices) > 0:
        first_future_trough_index = future_troughs_indices[0]
        corresponding_trough = ppg_troughs[first_future_trough_index]
        ptt_samples = corresponding_trough - r_peak
        ptt_ms = (ptt_samples / fs) * 1000
        
        if 50 < ptt_ms < 500:
            ptt_values_ms.append(ptt_ms)
            ptt_peak_times.append(r_peak / fs)

plt.figure(figsize=(17, 5))
plt.plot(ptt_peak_times, ptt_values_ms, marker='o', linestyle='-', color='green')
plt.title('time changing of PPt in time')
plt.xlabel('Time')
plt.ylabel('PTT ')
plt.grid(True)
plt.show()

print(f"\n Success")
print(f"Avrage of PTT {np.mean(ptt_values_ms):.2f} ms")