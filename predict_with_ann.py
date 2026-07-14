import os
import wfdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
import joblib

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs; low = lowcut / nyquist; high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band'); y = filtfilt(b, a, data)
    return y

def extract_features_from_new_file(base_file_name, folder_path):
    full_path = os.path.join(folder_path, base_file_name)
    print(f"--- Processing new file: {base_file_name} ---")
    try:
        record = wfdb.rdrecord(full_path); fs = record.fs
        ecg_index = record.sig_name.index('II'); ppg_index = record.sig_name.index('PLETH')
        ecg_signal_raw = record.p_signal[:, ecg_index]; ppg_signal_raw = record.p_signal[:, ppg_index]
        ecg_filtered = bandpass_filter(ecg_signal_raw, 0.5, 40.0, fs); ppg_filtered = bandpass_filter(ppg_signal_raw, 0.5, 4.0, fs)
        ecg_peaks, _ = find_peaks(ecg_filtered, height=np.mean(ecg_filtered), distance=int(0.5*fs))
        ppg_troughs, _ = find_peaks(-ppg_filtered, height=-np.mean(ppg_filtered), distance=int(0.5*fs))
        ptt_values_ms = []; hr_values = []; event_samples = []
        for i in range(1, len(ecg_peaks)):
            r_peak = ecg_peaks[i]; rr_interval_sec = (r_peak - ecg_peaks[i-1]) / fs; heart_rate = 60.0 / rr_interval_sec
            future_troughs = ppg_troughs[ppg_troughs > r_peak]
            if len(future_troughs) > 0 and 40 < heart_rate < 180:
                ptt_samples = future_troughs[0] - r_peak; ptt_ms = (ptt_samples / fs) * 1000
                if 50 < ptt_ms < 600:
                    ptt_values_ms.append(ptt_ms); hr_values.append(heart_rate); event_samples.append(r_peak)
        print(f"Feature extraction complete. Found {len(ptt_values_ms)} beats.")
        return ptt_values_ms, hr_values, event_samples, ecg_signal_raw, ppg_signal_raw, fs, record
    except Exception as e:
        print(f"Could not process file {base_file_name}. Error: {e}"); 
        return [], [], [], None, None, None, None

if __name__ == "__main__":
    print("Loading trained ANN models and parameters...")
    sbp_model = joblib.load('sbp_model_ann.joblib')
    dbp_model = joblib.load('dbp_model_ann.joblib')
    norm_params = joblib.load('norm_params_ann.joblib')
    print("Models loaded successfully.")

    folder_path = "D:/New folder (6)"
    new_file_name = "3042143_0013" 
    
    ptt_new, hr_new, samples_new, ecg_wave, ppg_wave, fs_new , record= extract_features_from_new_file(new_file_name, folder_path)

    if len(ptt_new) > 0:
        ptt_norm = (np.array(ptt_new) - norm_params['ptt_mean']) / norm_params['ptt_std']
        hr_norm = (np.array(hr_new) - norm_params['hr_mean']) / norm_params['hr_std']
        X_predict = np.column_stack((ptt_norm, hr_norm))
        
        sbp_predicted = sbp_model.predict(X_predict)
        dbp_predicted = dbp_model.predict(X_predict)
        
        plt.figure(figsize=(18, 9))
        ax1 = plt.subplot(2, 1, 1)
        ax1.plot(ecg_wave, label='Input ECG Signal'); ax1.set_title('Input Signals'); ax1.legend(); ax1.grid(True)
        ax2 = ax1.twinx()
        ax2.plot(ppg_wave, label='Input PPG Signal', color='red', alpha=0.6); ax2.legend(loc='lower right')
        
        ax3 = plt.subplot(2, 1, 2, sharex=ax1)
        ax3.plot(samples_new, sbp_predicted, '*', color='magenta', markersize=8, label='Predicted SBP (ANN)')
        ax3.plot(samples_new, dbp_predicted, '*', color='lime', markersize=8, label='Predicted DBP (ANN)')
        ax3.set_title('Predicted Blood Pressure Envelope'); ax3.set_xlabel('Sample Number'); ax3.set_ylabel('Predicted Blood Pressure (mmHg)'); ax3.legend(); ax3.grid(True)
        
        plt.tight_layout()
        plt.show()