import os
import wfdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
from sklearn.ensemble import RandomForestRegressor

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs; low = lowcut / nyquist; high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band'); y = filtfilt(b, a, data)
    return y

def process_record(base_file_name, folder_path):
    full_path = os.path.join(folder_path, base_file_name)
    print(f"--- Processing record: {base_file_name} ---")
    
    try:
        record = wfdb.rdrecord(full_path); fs = record.fs
        ecg_index = record.sig_name.index('II'); ppg_index = record.sig_name.index('PLETH'); abp_index = record.sig_name.index('ABP')
        ecg_signal_raw = record.p_signal[:, ecg_index]; ppg_signal_raw = record.p_signal[:, ppg_index]; abp_signal_raw = record.p_signal[:, abp_index]
        ecg_filtered = bandpass_filter(ecg_signal_raw, 0.5, 40.0, fs)
        ppg_filtered = bandpass_filter(ppg_signal_raw, 0.5, 4.0, fs)
        abp_filtered = bandpass_filter(abp_signal_raw, 0.5, 30.0, fs)

        ecg_peaks, _ = find_peaks(ecg_filtered, height=np.mean(ecg_filtered), distance=int(0.5*fs))
        ppg_troughs, _ = find_peaks(-ppg_filtered, height=-np.mean(ppg_filtered), distance=int(0.5*fs))
        sbp_peaks, _ = find_peaks(abp_filtered, height=np.mean(abp_filtered), distance=int(0.5*fs))
        dbp_peaks, _ = find_peaks(-abp_filtered, height=-np.mean(abp_filtered), distance=int(0.5*fs))
        
        sbp_values = abp_signal_raw[sbp_peaks]; dbp_values = abp_signal_raw[dbp_peaks]
        correlated_ptt = []; correlated_hr = []; correlated_sbp = []; correlated_dbp = []
        correlated_event_samples = []

        for i in range(1, len(ecg_peaks)):
            r_peak = ecg_peaks[i]
            rr_interval_sec = (r_peak - ecg_peaks[i-1]) / fs
            heart_rate = 60.0 / rr_interval_sec
            
            future_troughs = ppg_troughs[ppg_troughs > r_peak]
            if len(future_troughs) > 0 and 40 < heart_rate < 180:
                ptt_samples = future_troughs[0] - r_peak; ptt_ms = (ptt_samples / fs) * 1000
                if 50 < ptt_ms < 600:
                    ptt_time_sec = r_peak / fs
                    s_time_diffs = np.abs((sbp_peaks / fs) - ptt_time_sec); d_time_diffs = np.abs((dbp_peaks / fs) - ptt_time_sec)
                    if np.min(s_time_diffs) < 0.5 and np.min(d_time_diffs) < 0.5:
                        closest_sbp_index = np.argmin(s_time_diffs); closest_dbp_index = np.argmin(d_time_diffs)
                        correlated_ptt.append(ptt_ms); correlated_hr.append(heart_rate)
                        correlated_sbp.append(sbp_values[closest_sbp_index]); correlated_dbp.append(dbp_values[closest_dbp_index])
                        correlated_event_samples.append(r_peak)
        if len(correlated_ptt) > 1:
            ptt_array = np.array(correlated_ptt)
            hr_array = np.array(correlated_hr)
            ptt_mean, ptt_std = np.mean(ptt_array), np.std(ptt_array)
            hr_mean, hr_std = np.mean(hr_array), np.std(hr_array)
            if ptt_std == 0: ptt_std = 1
            if hr_std == 0: hr_std = 1
            ptt_normalized = (ptt_array - ptt_mean) / ptt_std
            hr_normalized = (hr_array - hr_mean) / hr_std
            
            print(f"Successfully processed and normalized. Found {len(ptt_normalized)} valid data points.")
            return ptt_normalized, hr_normalized, correlated_sbp, correlated_dbp, abp_signal_raw, correlated_event_samples, fs
        else:
            print("Not enough data to normalize. Skipping record.")
            return [], [], [], [], None, [], None

    except Exception as e:
        print(f"Could not process record {base_file_name}. Error: {e}")
        return [], [], [], [], None, [], None

if __name__ == "__main__":
    folder_path = "D:/New folder (6)"
    record_list = [ 
                   "3510174_0003",
        "3510174_0002",
        "3403274_0077",
        "3403274_0076",
        "3403274_0027",
        "3229745_0016",
        "3132009_0007",
        "3132009_0006"
                   ]
    all_ptt_norm = []; all_hr_norm = []; all_sbp = []; all_dbp = []
    
    for record_name in record_list:
        ptt, hr, sbp, dbp, _, _, _ = process_record(record_name, folder_path)
        if len(ptt) > 0:
            all_ptt_norm.extend(ptt); all_hr_norm.extend(hr); all_sbp.extend(sbp); all_dbp.extend(dbp)
            
    print("\n==============================================")
    print(f"Total data points for training: {len(all_ptt_norm)}")
    X_train = np.column_stack((all_ptt_norm, all_hr_norm))
    y_sbp_train = np.array(all_sbp)
    y_dbp_train = np.array(all_dbp)
    print("Training the final models on NORMALIZED data...")
    sbp_model = RandomForestRegressor(n_estimators=100, random_state=42); sbp_model.fit(X_train, y_sbp_train)
    dbp_model = RandomForestRegressor(n_estimators=100, random_state=42); dbp_model.fit(X_train, y_dbp_train)
    print("Models trained successfully.")
    print("\nEvaluating model on a sample record...")
    test_record_name = "3229745_0016"
    ptt_test_norm, hr_test_norm, sbp_actual, dbp_actual, abp_wave_test, event_samples_test, fs_test = process_record(test_record_name, folder_path)
    
    if len(ptt_test_norm) > 0:
        X_test = np.column_stack((ptt_test_norm, hr_test_norm))
        sbp_predicted = sbp_model.predict(X_test)
        dbp_predicted = dbp_model.predict(X_test)
        plt.figure(figsize=(18, 8))
        plt.plot(abp_wave_test, color='gray', alpha=0.5, label='Actual ABP Waveform')
        plt.plot(event_samples_test, sbp_actual, 'o', color='blue', markersize=8, label='Actual SBP')
        plt.plot(event_samples_test, dbp_actual, 'o', color='blue', markersize=8, label='Actual DBP')
        plt.plot(event_samples_test, sbp_predicted, '*', color='red', markersize=10, label='Predicted SBP (Normalized Model)')
        plt.plot(event_samples_test, dbp_predicted, '*', color='gold', markersize=10, label='Predicted DBP (Normalized Model)')

        plt.title('Final Model Evaluation with Normalized Data')
        plt.xlabel('Sample Number'); plt.ylabel('Blood Pressure (mmHg)')
        plt.legend(); plt.grid(True)
        plt.xlim(12500, 14000)
        plt.show()