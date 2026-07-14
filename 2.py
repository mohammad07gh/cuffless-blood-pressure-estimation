import os
import wfdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks

folder_path = "D:/New folder (6)"  
base_file_name = "3229745_0016"  
full_path = os.path.join(folder_path, base_file_name)
print(f"Attempting to read record: {full_path}")

try:
    record = wfdb.rdrecord(full_path)
    fs = record.fs
    print(f"Record read successfully. Sampling Frequency: {fs} Hz")
    ecg_index = record.sig_name.index('II')
    ppg_index = record.sig_name.index('PLETH')
    abp_index = record.sig_name.index('ABP')
    print("All required signals ('II', 'PLETH', 'ABP') found in the record.")
    ecg_signal_raw = record.p_signal[:, ecg_index]
    ppg_signal_raw = record.p_signal[:, ppg_index]
    abp_signal_raw = record.p_signal[:, abp_index]

except Exception as e:
    print(f"An error occurred: {e}")
    exit()

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y
ecg_filtered = bandpass_filter(ecg_signal_raw, lowcut=0.5, highcut=40.0, fs=fs)
ppg_filtered = bandpass_filter(ppg_signal_raw, lowcut=0.5, highcut=4.0, fs=fs)
abp_filtered = bandpass_filter(abp_signal_raw, lowcut=0.5, highcut=30.0, fs=fs)

fig, axs = plt.subplots(3, 1, figsize=(18, 10), sharex=True)
num_samples_to_plot = 5 * fs  # نمایش ۵ ثانیه

# ECG Plot
axs[0].plot(ecg_signal_raw[:num_samples_to_plot], label='Raw ECG', color='silver')
axs[0].plot(ecg_filtered[:num_samples_to_plot], label='Filtered ECG', color='blue')
axs[0].set_title('ECG Signal Filtering')
axs[0].legend()
axs[0].grid(True)

# PPG Plot
axs[1].plot(ppg_signal_raw[:num_samples_to_plot], label='Raw PPG', color='silver')
axs[1].plot(ppg_filtered[:num_samples_to_plot], label='Filtered PPG', color='red')
axs[1].set_title('PPG Signal Filtering')
axs[1].legend()
axs[1].grid(True)

# ABP Plot
axs[2].plot(abp_signal_raw[:num_samples_to_plot], label='Raw ABP', color='silver')
axs[2].plot(abp_filtered[:num_samples_to_plot], label='Filtered ABP', color='purple')
axs[2].set_title('ABP Signal Filtering')
axs[2].legend()
axs[2].grid(True)

plt.xlabel('Sample Number')
plt.tight_layout()
plt.show()


ecg_peak_distance = int(0.5 * fs) # حداقل فاصله معادل 0.5 ثانیه
ecg_peaks, _ = find_peaks(ecg_filtered, height=np.mean(ecg_filtered), distance=ecg_peak_distance)


ppg_trough_distance = int(0.5 * fs)
ppg_troughs, _ = find_peaks(-ppg_filtered, height=-np.mean(ppg_filtered), distance=ppg_trough_distance)

# --- محاسبه PTT ---
ptt_values_ms = []
ptt_peak_times_sec = []

for r_peak in ecg_peaks:
    future_troughs = ppg_troughs[ppg_troughs > r_peak]
    if len(future_troughs) > 0:
        corresponding_trough = future_troughs[0]
        ptt_samples = corresponding_trough - r_peak
        ptt_ms = (ptt_samples / fs) * 1000
        
        if 50 < ptt_ms < 600: 
            ptt_values_ms.append(ptt_ms)
            ptt_peak_times_sec.append(r_peak / fs)

plt.figure(figsize=(18, 5))
plt.plot(ptt_peak_times_sec, ptt_values_ms, marker='o', linestyle='-', color='green')
plt.title('Pulse Transit Time (PTT) Variation Over Time')
plt.xlabel('Time (seconds)')
plt.ylabel('PTT (milliseconds)')
plt.grid(True)
plt.show()

print(f"PTT calculation complete. Found {len(ptt_values_ms)} valid PTT values.")
print(f"Average PTT: {np.mean(ptt_values_ms):.2f} ms")



abp_peak_distance = int(0.5 * fs)
sbp_peaks, _ = find_peaks(abp_filtered, height=np.mean(abp_filtered), distance=abp_peak_distance)
dbp_peaks, _ = find_peaks(-abp_filtered, height=-np.mean(abp_filtered), distance=abp_peak_distance)

# استخراج مقادیر فشار خون سیستولیک 
sbp_values = abp_signal_raw[sbp_peaks]

# پیدا کردن مقدار فشار خون سیستولیک متناظر با هر PTT
correlated_sbp = []
correlated_ptt = []

for i, ptt_time in enumerate(ptt_peak_times_sec):
    time_diffs = np.abs((sbp_peaks / fs) - ptt_time)
    if np.min(time_diffs) < 0.5:
        closest_sbp_index = np.argmin(time_diffs)
        correlated_sbp.append(sbp_values[closest_sbp_index])
        correlated_ptt.append(ptt_values_ms[i])


plt.figure(figsize=(10, 6))
plt.scatter(correlated_ptt, correlated_sbp, alpha=0.6, color='purple')
plt.title('Correlation between PTT and Systolic Blood Pressure (SBP)')
plt.xlabel('Pulse Transit Time (ms)')
plt.ylabel('Systolic Blood Pressure (mmHg)')
plt.grid(True)
plt.show()

print("\nProject Complete!")