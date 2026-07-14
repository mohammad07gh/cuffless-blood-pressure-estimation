import os
import wfdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
from sklearn.ensemble import RandomForestRegressor

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

abp_peak_distance = int(0.5 * fs)
sbp_peaks, _ = find_peaks(abp_filtered, height=np.mean(abp_filtered), distance=abp_peak_distance)
sbp_values = abp_signal_raw[sbp_peaks]
dbp_peaks, _ = find_peaks(-abp_filtered, height=-np.mean(abp_filtered), distance=abp_peak_distance)
dbp_values = abp_signal_raw[dbp_peaks]

correlated_sbp = []; correlated_dbp = []; correlated_ptt = []; correlated_ptt_times = []
for i, ptt_time in enumerate(ptt_peak_times_sec):
    s_time_diffs = np.abs((sbp_peaks / fs) - ptt_time)
    d_time_diffs = np.abs((dbp_peaks / fs) - ptt_time)
    if np.min(s_time_diffs) < 0.5 and np.min(d_time_diffs) < 0.5:
        closest_sbp_index = np.argmin(s_time_diffs)
        closest_dbp_index = np.argmin(d_time_diffs)
        correlated_sbp.append(sbp_values[closest_sbp_index]); correlated_dbp.append(dbp_values[closest_dbp_index])
        correlated_ptt.append(ptt_values_ms[i]); correlated_ptt_times.append(ptt_time)

print("\nBuilding separate models for SBP and DBP...")
X = np.array(correlated_ptt).reshape(-1, 1)
y_sbp = np.array(correlated_sbp)
y_dbp = np.array(correlated_dbp)

sbp_model = RandomForestRegressor(n_estimators=100, random_state=42); sbp_model.fit(X, y_sbp)
sbp_predicted = sbp_model.predict(X)
dbp_model = RandomForestRegressor(n_estimators=100, random_state=42); dbp_model.fit(X, y_dbp)
dbp_predicted = dbp_model.predict(X)

print("\nGenerating final comparison plot...")

plt.figure(figsize=(18, 8))

# --- رسم موج فشار خون واقعی (ABP) ---
plt.plot(abp_signal_raw, color='gray', alpha=0.5, label='Actual ABP Waveform')

# --- تبدیل زمان به شماره نمونه برای هماهنگی ---
correlated_event_samples = (np.array(correlated_ptt_times) * fs).astype(int)

# --- **تغییر کلیدی:** اضافه کردن نقاط واقعی (دایره آبی) ---
plt.plot(correlated_event_samples, y_sbp, 'o', color='blue', markersize=8, label='Actual SBP')
plt.plot(correlated_event_samples, y_dbp, 'o', color='blue', markersize=8, label='Actual DBP')

# --- رسم پیش‌بینی‌ها به صورت ستاره ---
plt.plot(correlated_event_samples, sbp_predicted, '*', color='red', markersize=10, label='Predicted SBP')
plt.plot(correlated_event_samples, dbp_predicted, '*', color='gold', markersize=10, label='Predicted DBP')

plt.title('Final Comparison: Actual vs. Predicted Blood Pressure')
plt.xlabel('Sample Number')
plt.ylabel('Blood Pressure (mmHg)')
plt.legend()
plt.grid(True)
plt.xlim(12500, 14000) # محدود کردن محور افقی برای نمایش بهتر
plt.show()

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

mae_sbp = mean_absolute_error(y_sbp, sbp_predicted)
print(f"Mean Absolute Error (MAE) for SBP: {mae_sbp:.2f} mmHg")

rmse_sbp = np.sqrt(mean_squared_error(y_sbp, sbp_predicted))
print(f"Root Mean Squared Error (RMSE) for SBP: {rmse_sbp:.2f} mmHg")

r2_sbp = r2_score(y_sbp, sbp_predicted)
print(f"R-squared (R2) for SBP: {r2_sbp:.2f}")

mae_dbp = mean_absolute_error(y_dbp, dbp_predicted)
print(f"Mean Absolute Error (MAE) for DBP: {mae_sbp:.2f} mmHg")
rmse_dbp = np.sqrt(mean_squared_error(y_dbp, dbp_predicted))
print(f"Root Mean Squared Error (RMSE) for DBP: {rmse_sbp:.2f} mmHg")
r2_dbp = r2_score(y_dbp, dbp_predicted)
print(f"R-squared (R2) for DBP: {r2_sbp:.2f}")