import wfdb
import os
import matplotlib.pyplot as plt

# --- فقط این دو خط را در صورت نیاز تغییر دهید ---
folder_path = "D:/New folder (6)"
file_to_check = "3042143_0013"
# ---------------------------------------------

full_path = os.path.join(folder_path, file_to_check)

try:
    record = wfdb.rdrecord(full_path)
    ecg = record.p_signal[:, record.sig_name.index('II')]
    ppg = record.p_signal[:, record.sig_name.index('PLETH')]
    
    print(f"Plotting raw signals for {file_to_check}...")
    
    fig, axs = plt.subplots(2, 1, figsize=(18, 8), sharex=True)
    axs[0].plot(ecg)
    axs[0].set_title('Raw ECG Signal (II)')
    axs[0].grid(True)
    
    axs[1].plot(ppg, color='red')
    axs[1].set_title('Raw PPG Signal (PLETH)')
    axs[1].grid(True)
    
    plt.xlabel('Sample Number')
    plt.show()

except Exception as e:
    print(f"An error occurred while reading or plotting: {e}")