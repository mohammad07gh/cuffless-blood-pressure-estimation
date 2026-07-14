import os
import wfdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

folder_path = "D:/New folder (6)"
base_file_name = "3229745_0016"

full_path = os.path.join(folder_path, base_file_name)
print(f"Attempting to read record: {full_path}")
try:
    record = wfdb.rdrecord(full_path); fs = record.fs
    print(f"Record read successfully. Sampling Frequency: {fs} Hz")
    ecg_index = record.sig_name.index('II'); ppg_index = record.sig_name.index('PLETH'); abp_index = record.sig_name.index('ABP')
    print("All required signals ('II', 'PLETH', 'ABP') found.")
    ecg_signal_raw = record.p_signal[:, ecg_index]; ppg_signal_raw = record.p_signal[:, ppg_index]; abp_signal_raw = record.p_signal[:, abp_index]
except Exception as e:
    print(f"An error occurred: {e}"); exit()

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs; low = lowcut / nyquist; high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band'); y = filtfilt(b, a, data)
    return y
ecg_filtered = bandpass_filter(ecg_signal_raw, 0.5, 40.0, fs); ppg_filtered = bandpass_filter(ppg_signal_raw, 0.5, 4.0, fs); abp_filtered = bandpass_filter(abp_signal_raw, 0.5, 30.0, fs)

ecg_peaks, _ = find_peaks(ecg_filtered, height=np.mean(ecg_filtered), distance=int(0.5*fs))
ppg_troughs, _ = find_peaks(-ppg_filtered, height=-np.mean(ppg_filtered), distance=int(0.5*fs))
ptt_values_ms = []; ptt_peak_times_sec = []
for r_peak in ecg_peaks:
    future_troughs = ppg_troughs[ppg_troughs > r_peak]
    if len(future_troughs) > 0:
        ptt_samples = future_troughs[0] - r_peak; ptt_ms = (ptt_samples / fs) * 1000
        if 50 < ptt_ms < 600:
            ptt_values_ms.append(ptt_ms); ptt_peak_times_sec.append(r_peak / fs)

sbp_peaks, _ = find_peaks(abp_filtered, height=np.mean(abp_filtered), distance=int(0.5*fs))
sbp_values = abp_signal_raw[sbp_peaks]
correlated_sbp = []; correlated_ptt = []; correlated_ptt_times = []
for i, ptt_time in enumerate(ptt_peak_times_sec):
    time_diffs = np.abs((sbp_peaks / fs) - ptt_time)
    if np.min(time_diffs) < 0.5:
        closest_sbp_index = np.argmin(time_diffs)
        correlated_sbp.append(sbp_values[closest_sbp_index]); correlated_ptt.append(ptt_values_ms[i]); correlated_ptt_times.append(ptt_time)

print("\nBuilding the Polynomial Prediction Model...")

X = np.array(correlated_ptt).reshape(-1, 1)
y = np.array(correlated_sbp)

poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)

poly_model = LinearRegression()
poly_model.fit(X_poly, y)
sbp_predicted_poly = poly_model.predict(X_poly)

print("\nGenerating final time-series comparison plot...")

start_time_sec = 100
end_time_sec = start_time_sec + 10
start_sample = int(start_time_sec * fs)
end_sample = int(end_time_sec * fs)
time_axis_segment = np.arange(start_sample, end_sample) / fs

plt.figure(figsize=(18, 7))
plt.plot(time_axis_segment, abp_signal_raw[start_sample:end_sample], color='gray', alpha=0.8, label='Actual ABP Waveform')

indices_in_segment = [i for i, t in enumerate(correlated_ptt_times) if start_time_sec < t < end_time_sec]

if indices_in_segment:
    times_in_segment = [correlated_ptt_times[i] for i in indices_in_segment]
    actual_sbp_in_segment = [y[i] for i in indices_in_segment]
    predicted_sbp_in_segment = [sbp_predicted_poly[i] for i in indices_in_segment]
    
    plt.plot(times_in_segment, actual_sbp_in_segment, 'o', color='blue', markersize=10, label='Actual SBP')
    plt.plot(times_in_segment, predicted_sbp_in_segment, '*', color='green', markersize=12, label='Predicted SBP (Polynomial Model)')

plt.title('Time-Series Comparison: Actual vs. Non-Linear Model Prediction')
plt.xlabel('Time (seconds)')
plt.ylabel('Blood Pressure (mmHg)')
plt.legend()
plt.grid(True)
plt.show()