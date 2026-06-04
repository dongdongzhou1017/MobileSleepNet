import numpy as np
import scipy.io as scio
from os import path
from scipy import signal
from scipy.signal import butter, lfilter, iirnotch

path_Extracted = './datasets/ISRUC/ISRUC_S3/ExtractedChannels/'
path_RawData   = './datasets/ISRUC/ISRUC_S3/RawData/'
path_output    = './datasets/ISRUC/ISRUC_S3/'
channels = ['C3_A2', 'C4_A1', 'F3_A2', 'F4_A1', 'O1_A2', 'O2_A1',
            'LOC_A2', 'ROC_A1','X1', 'X2']


def read_psg(path_Extracted, sub_id, channels, resample=3000):
    psg = scio.loadmat(path.join(path_Extracted, 'subject%d.mat' % (sub_id)))
    psg_use = []
    for c in channels:
        psg_use.append(
            np.expand_dims(signal.resample(psg[c], resample, axis=-1), 1))
    psg_use = np.concatenate(psg_use, axis=1)
    return psg_use


def read_label(path_RawData, sub_id, ignore=30):
    label = []
    with open(path.join(path_RawData, '%d/%d_1.txt' % (sub_id, sub_id))) as f:
        s = f.readline()
        while True:
            a = s.replace('\n', '')
            label.append(int(a))
            s = f.readline()
            if s == '' or s == '\n':
                break
    return np.array(label[:-ignore])


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def notch_filter(data, freq, fs, Q=30):
    nyq = 0.5 * fs
    w0 = freq / nyq
    b, a = iirnotch(w0, Q)
    y = lfilter(b, a, data)
    return y


fs = 100  
eeg_lowcut = 0.3
eeg_highcut = 35

fold_label = []
fold_psg = []
fold_len = []

for sub in range(1, 11):
    print('Read subject', sub)
    label = read_label(path_RawData, sub)
    psg = read_psg(path_Extracted, sub, channels)
    print('Subject', sub, ':', label.shape, psg.shape)
    assert len(label) == len(psg)

    # in ISRUC, 0-Wake, 1-N1, 2-N2, 3-N3, 5-REM
    label[label == 5] = 4  # make 4 correspond to REM
    fold_label.append(np.eye(5)[label])
    

    eeg_channels = ['C3_A2', 'C4_A1', 'F3_A2', 'F4_A1', 'O1_A2', 'O2_A1']
    eog_channels = ['LOC_A2', 'ROC_A1']
    emg_channels = ['X1', 'X2']
    
    eeg_data = psg[:, [channels.index(c) for c in eeg_channels]]
    eog_data = psg[:, [channels.index(c) for c in eog_channels]]
    emg_data = psg[:, [channels.index(c) for c in emg_channels]]
    
    eeg_data = butter_bandpass_filter(eeg_data, eeg_lowcut, eeg_highcut, fs)

    # Combine preprocessed channels
    psg_preprocessed = np.concatenate([eeg_data, eog_data, emg_data], axis=1)
    
    # Reshape data to (num_segments, channels, samples_per_segment)
    num_segments = len(label)
    psg_preprocessed = psg_preprocessed.reshape(num_segments, len(channels), -1)

    
    # Combine the frequency bands with the other channels
    psg_preprocessed = np.concatenate([
        psg_preprocessed,
        psg[:, [channels.index('C3_A2'), channels.index('C4_A1'), channels.index('F3_A2'), channels.index('F4_A1'),
                channels.index('O1_A2'), channels.index('O2_A1'), channels.index('LOC_A2'), channels.index('ROC_A1'),
                channels.index('X1'), channels.index('X2')]]
    ], axis=1)
    
    print(psg_preprocessed.shape)
    fold_psg.append(psg_preprocessed)
    fold_len.append(len(label))

print('Preprocess over.')

np.savez(path.join(path_output, 'ISRUC_S3_all.npz'),
    Fold_data=np.array(fold_psg, dtype=object),
    Fold_label=np.array(fold_label, dtype=object),
    Fold_len=np.array(fold_len, dtype=object)
)
print('Saved to', path.join(path_output, 'ISRUC_S3_all.npz'))
