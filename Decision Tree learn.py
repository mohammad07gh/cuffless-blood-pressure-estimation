import os
import wfdb
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from sklearn.tree import DecisionTreeRegressor
import joblib # برای ذخیره مدل

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs; low = lowcut / nyquist; high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band'); y = filtfilt(b, a, data)
    return y

def process_record_for_training(base_file_name, folder_path):
    full_path = os.path.join(folder_path, base_file_name)
    print(f"--- Processing record: {base_file_name} ---")
    try:
        record = wfdb.rdrecord(full_path); fs = record.fs
        ecg_index = record.sig_name.index('II'); ppg_index = record.sig_name.index('PLETH'); abp_index = record.sig_name.index('ABP')
        ecg_signal_raw = record.p_signal[:, ecg_index]; ppg_signal_raw = record.p_signal[:, ppg_index]; abp_signal_raw = record.p_signal[:, abp_index]
        ecg_filtered = bandpass_filter(ecg_signal_raw, 0.5, 40.0, fs); ppg_filtered = bandpass_filter(ppg_signal_raw, 0.5, 4.0, fs); abp_filtered = bandpass_filter(abp_signal_raw, 0.5, 30.0, fs)
        ecg_peaks, _ = find_peaks(ecg_filtered, height=np.mean(ecg_filtered), distance=int(0.5*fs))
        ppg_troughs, _ = find_peaks(-ppg_filtered, height=-np.mean(ppg_filtered), distance=int(0.5*fs))
        sbp_peaks, _ = find_peaks(abp_filtered, height=np.mean(abp_filtered), distance=int(0.5*fs)); dbp_peaks, _ = find_peaks(-abp_filtered, height=-np.mean(abp_filtered), distance=int(0.5*fs))
        sbp_values = abp_signal_raw[sbp_peaks]; dbp_values = abp_signal_raw[dbp_peaks]
        correlated_ptt = []; correlated_hr = []; correlated_sbp = []; correlated_dbp = []
        for i in range(1, len(ecg_peaks)):
            r_peak = ecg_peaks[i]; rr_interval_sec = (r_peak - ecg_peaks[i-1]) / fs; heart_rate = 60.0 / rr_interval_sec
            future_troughs = ppg_troughs[ppg_troughs > r_peak]
            if len(future_troughs) > 0 and 40 < heart_rate < 180:
                ptt_samples = future_troughs[0] - r_peak; ptt_ms = (ptt_samples / fs) * 1000
                if 50 < ptt_ms < 600:
                    ptt_time_sec = r_peak / fs
                    s_time_diffs = np.abs((sbp_peaks / fs) - ptt_time_sec); d_time_diffs = np.abs((dbp_peaks / fs) - ptt_time_sec)
                    if np.min(s_time_diffs) < 0.5 and np.min(d_time_diffs) < 0.5:
                        closest_sbp_index = np.argmin(s_time_diffs); closest_dbp_index = np.argmin(d_time_diffs)
                        correlated_ptt.append(ptt_ms); correlated_hr.append(heart_rate); correlated_sbp.append(sbp_values[closest_sbp_index]); correlated_dbp.append(dbp_values[closest_dbp_index])
        print(f"Successfully processed. Found {len(correlated_ptt)} data points.")
        return correlated_ptt, correlated_hr, correlated_sbp, correlated_dbp
    except Exception as e:
        print(f"Could not process record {base_file_name}. Error: {e}"); return [], [], [], []

if __name__ == "__main__":
    folder_path = "D:/New folder (6)"
    record_list = [ "3403274_0077", "3403274_0076", "3403274_0027", "3229745_0016", "3132009_0007" ]
    all_ptt = []; all_hr = []; all_sbp = []; all_dbp = []
    for record_name in record_list:
        ptt, hr, sbp, dbp = process_record_for_training(record_name, folder_path)
        if len(ptt) > 0:
            all_ptt.extend(ptt); all_hr.extend(hr); all_sbp.extend(sbp); all_dbp.extend(dbp)
    print(f"\nTotal data points for training: {len(all_ptt)}")
    
    ptt_mean = np.mean(all_ptt); ptt_std = np.std(all_ptt)
    hr_mean = np.mean(all_hr); hr_std = np.std(all_hr)
    all_ptt_norm = (np.array(all_ptt) - ptt_mean) / ptt_std
    all_hr_norm = (np.array(all_hr) - hr_mean) / hr_std
    X_train = np.column_stack((all_ptt_norm, all_hr_norm))
    y_sbp_train = np.array(all_sbp); y_dbp_train = np.array(all_dbp)
    
    print("Training Decision Tree models...")
    sbp_model = DecisionTreeRegressor(random_state=42); sbp_model.fit(X_train, y_sbp_train)
    dbp_model = DecisionTreeRegressor(random_state=42); dbp_model.fit(X_train, y_dbp_train)
    print("Models trained successfully.")

    print("\nSaving models and normalization parameters...")
    joblib.dump(sbp_model, 'sbp_model_dt.joblib')
    joblib.dump(dbp_model, 'dbp_model_dt.joblib')
    norm_params = {'ptt_mean': ptt_mean, 'ptt_std': ptt_std, 'hr_mean': hr_mean, 'hr_std': hr_std}
    joblib.dump(norm_params, 'norm_params_dt.joblib')
    print("Files saved successfully.")