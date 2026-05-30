import numpy as np
# sys.path.append('../input')

import os
import time
import psutil
import tensorflow as tf
from scipy.signal import savgol_filter
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix
from tensorflow import keras
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Flatten, MaxPooling1D, Activation,\
BatchNormalization, Add, Reshape, TimeDistributed, Input, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K




from Utils import *
from DataGenerator import *




Path = "./datasets/sleepedf-2013/npz/sleep_edf_processed_Fpz.npz"
output_path = "./MFE/output_sleepedf2013/testlightsingle/"



ReadList = np.load(Path, allow_pickle=True)
Fold_Num   = ReadList['Fold_len']    # Num of samples of each fold
Fold_Data  = ReadList['Fold_data']   # Data of each fold
Fold_Label = ReadList['Fold_label']  # Labels of each fold


freq = 100
channels = 1
subject_num = len(Fold_Num)
fold = 10


HMM_EMISSION_CONFIG = {
    'max_samples': 5000,          
    'sampling_strategy': 'random',  
    'enable_sampling': True       
}

cfg = {
    'bs': 32,
    'epochs': 50
}

DataGenerator = kFoldGenerator(Fold_Data, Fold_Label, fold, subject_num)
del ReadList, Fold_Label

print('y_list length:', len(DataGenerator.y_list))

if not os.path.exists(output_path):
    os.makedirs(output_path)


from sklearn import metrics

import random as python_random
os.environ['TF_DETERMINISTIC_OPS'] = '1'
np.random.seed(32)
python_random.seed(32)
tf.random.set_seed(32)
print("keras version:", keras.__version__)
 
# print(device_lib.list_local_devices())

tf.config.set_soft_device_placement(True)


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Exclude GPU:2 and set GPU:1 as the visible device
        tf.config.experimental.set_visible_devices(gpus[1], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[1], True)
        print("Using GPU:1 for computations")
    except RuntimeError as e:
        print(e)
else:
    print("No GPUs available. Using CPU for computations.")


print("Current device:", tf.config.get_visible_devices('GPU'))



from tensorflow.keras.layers import SeparableConv1D

def CNN_light(inputs, fs=8, kernel_size=3, pool_size=2, weight=0.001):
    x = SeparableConv1D(fs, kernel_size, 1, padding='same', kernel_regularizer=l2(weight))(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size, 2, padding='same')(x)
    return x

def ResNet_light(inputs, fs=8, ks_1=3, ps_1=2, ks_2=3, ps_2=2, weight=0.001):
    x = CNN_light(inputs, fs, ks_1, ps_1, weight)
    x = CNN_light(x, fs, ks_2, ps_2, weight)
    shortcut_x = SeparableConv1D(fs, 1, 2, padding='same')(inputs)
    shortcut_x = SeparableConv1D(fs, 1, 2, padding='same')(shortcut_x)
    return Add()([x, shortcut_x])

def create_model_light(input_shape, channels=10, time_second=30, freq=100):
    inputs_channel = Input(shape=(time_second*freq, 1))
    x = ResNet_light(inputs_channel, 8)
    x = Dropout(0.1)(x)
    x = ResNet_light(x, 16)
    x = Dropout(0.1)(x)
    x = ResNet_light(x, 32)
    x = Dropout(0.1)(x)

    outputs = GlobalAveragePooling1D()(x)

    fea_part = Model(inputs=inputs_channel, outputs=outputs)
    inputs = Input(shape=input_shape)  # (3000, 10)
    input_re = Reshape((channels, time_second*freq, 1))(inputs)  # (10, 3000, 1)
    fea_all = TimeDistributed(fea_part)(input_re)

    fla_fea = Flatten()(fea_all)
    fla_fea = Dropout(0.3)(fla_fea)

    merged = Dense(64)(fla_fea)
    label_out = Dense(5, activation='softmax', name='Label')(merged)

    ce_model = Model(inputs, label_out)
    # ce_model.summary()
    return ce_model




def savgol_smooth_probabilities(model_probs, window_length=5, polyorder=2):

    if INFERENCE_OPTIMIZATION_CONFIG.get('use_fast_savgol', False):
        return savgol_smooth_probabilities_fast(model_probs, window_length, polyorder)
    
    n_samples, n_classes = model_probs.shape
    

    if window_length % 2 == 0:
        window_length += 1
    window_length = min(window_length, n_samples)
    

    polyorder = min(polyorder, window_length - 1)
    

    if n_samples < 3:
        return model_probs, np.argmax(model_probs, axis=1)
    

    smoothed_probs = np.zeros_like(model_probs)
    
    
    for i in range(model_probs.shape[1]):
        try:
            smoothed_probs[:, i] = savgol_filter(
                model_probs[:, i], 
                window_length=window_length, 
                polyorder=polyorder
            )
        except ValueError as e:
            if not INFERENCE_OPTIMIZATION_CONFIG.get('reduce_verbose', False):
                print(f"Savitzky-Golay filter parameter error (class {i}): {e}")
                print(f"Using original probabilities, window_length={window_length}, polyorder={polyorder}")
            smoothed_probs[:, i] = model_probs[:, i]
    

    smoothed_probs = np.clip(smoothed_probs, 0, 1)
    

    row_sums = smoothed_probs.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    smoothed_probs = smoothed_probs / row_sums
    

    smoothed_prediction = np.argmax(smoothed_probs, axis=1)
    
    return smoothed_probs, smoothed_prediction

def savgol_smooth_probabilities_fast(model_probs, window_length=5, polyorder=2):

    n_samples, n_classes = model_probs.shape
    

    if n_samples < 3:
        return model_probs, np.argmax(model_probs, axis=1)
    

    window_length = min(max(3, window_length | 1), n_samples)
    polyorder = min(polyorder, window_length - 1)
    

    dtype = np.float32 if INFERENCE_OPTIMIZATION_CONFIG.get('use_float32', False) else np.float64
    smoothed_probs = np.zeros_like(model_probs, dtype=dtype)
    

    try:
        for i in range(n_classes):
            smoothed_probs[:, i] = savgol_filter(
                model_probs[:, i].astype(dtype), 
                window_length=window_length,
                polyorder=polyorder
            )
    except ValueError:

        return model_probs, np.argmax(model_probs, axis=1)
    

    smoothed_probs = np.clip(smoothed_probs, 0, 1)
    row_sums = smoothed_probs.sum(axis=1, keepdims=True)
    np.divide(smoothed_probs, row_sums, out=smoothed_probs, where=row_sums!=0)
    
    return smoothed_probs, np.argmax(smoothed_probs, axis=1)

def map_labels_to_strings(numeric_labels):

    label_map = {0: 'W', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'REM'}
    return np.array([label_map[label] for label in numeric_labels])


class SleepStageHMM:

    def __init__(self, n_states=5):
        self.n_states = n_states
        self.state_names = ['W', 'N1', 'N2', 'N3', 'REM']
        self.transition_matrix = None
        self.emission_matrix = None
        self.initial_probs = None
        
    def estimate_transition_matrix_from_data(self, true_sequences):

        S = self.n_states
        A = np.zeros((S, S))
        

        total_transitions = 0
        for seq in true_sequences:
            if len(seq) < 2:
                continue
            for i in range(len(seq) - 1):
                from_state = int(seq[i])
                to_state = int(seq[i + 1])
                if 0 <= from_state < S and 0 <= to_state < S:
                    A[from_state, to_state] += 1
                    total_transitions += 1
        

        row_sums = A.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)  
        A = A / row_sums
        

        epsilon = 1e-8
        A = A + epsilon
        A = A / A.sum(axis=1, keepdims=True)
        
        self.transition_matrix = A
       
        return A
    
    
    
    def estimate_emission_matrix_from_confusion(self, true_labels, predicted_labels, max_samples=None, sampling_strategy='random'):

        S = self.n_states
        original_len = len(true_labels)
        

        if max_samples is not None and max_samples < original_len:
            true_labels, predicted_labels = self._sample_for_emission_estimation(
                true_labels, predicted_labels, max_samples, sampling_strategy
            )
            print(f"Sampling strategy: {sampling_strategy}, sampled {len(true_labels)} samples from {original_len} for emission matrix estimation")
        
        B = np.zeros((S, S))
        
        true_array = np.array(true_labels, dtype=int)
        pred_array = np.array(predicted_labels, dtype=int)
        
        valid_mask = (true_array >= 0) & (true_array < S) & (pred_array >= 0) & (pred_array < S)
        valid_true = true_array[valid_mask]
        valid_pred = pred_array[valid_mask]
        
        if len(valid_true) > 0:
            B = sklearn_confusion_matrix(valid_true, valid_pred, labels=list(range(S)))
        else:
            B = np.zeros((S, S))
        
        row_sums = B.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        B = B / row_sums
        
        epsilon = 1e-8
        B = B + epsilon
        B = B / B.sum(axis=1, keepdims=True)
        
        self.emission_matrix = B
        print(f"Pure data-driven emission matrix estimation from {len(true_labels)} samples")
        return B
    
    def _sample_for_emission_estimation(self, true_labels, predicted_labels, max_samples, strategy='random'):
        true_array = np.array(true_labels)
        pred_array = np.array(predicted_labels)
        n_samples = len(true_labels)
        
        if strategy == 'random':
            indices = np.random.choice(n_samples, size=max_samples, replace=False)
            
        elif strategy == 'stratified':
            indices = []
            for state in range(self.n_states):
                state_indices = np.where(true_array == state)[0]
                if len(state_indices) > 0:
                    state_ratio = len(state_indices) / n_samples
                    state_samples = int(max_samples * state_ratio)
                    state_samples = min(state_samples, len(state_indices))
                    
                    if state_samples > 0:
                        selected = np.random.choice(state_indices, size=state_samples, replace=False)
                        indices.extend(selected)
            
            indices = np.array(indices)
            
        elif strategy == 'balanced':
            samples_per_class = max_samples // self.n_states
            indices = []
            
            for state in range(self.n_states):
                state_indices = np.where(true_array == state)[0]
                if len(state_indices) > 0:
                    actual_samples = min(samples_per_class, len(state_indices))
                    selected = np.random.choice(state_indices, size=actual_samples, replace=False)
                    indices.extend(selected)
            
            indices = np.array(indices)
            
        else:
            raise ValueError(f"Unknown sampling strategy: {strategy}")
        
        return true_array[indices], pred_array[indices]
    
    def estimate_initial_probabilities_from_data(self, true_sequences):
        S = self.n_states
        
        self.initial_probs = np.zeros(S)
        self.initial_probs[0] = 1.0
        
        print(f"Using fixed initial probability vector: W=1.000, N1=0.000, N2=0.000, N3=0.000, REM=0.000")
        return self.initial_probs
    
    
    def viterbi_decode(self, observation_sequence):
        T = len(observation_sequence)
        
        if T == 0:
            return [], 0.0
        
        viterbi_table = np.zeros((T, self.n_states))
        path_table = np.zeros((T, self.n_states), dtype=int)
        
        obs_0 = observation_sequence[0]
        viterbi_table[0, :] = self.initial_probs * self.emission_matrix[:, obs_0]
        
        for t in range(1, T):
            obs_t = observation_sequence[t]
            for curr_state in range(self.n_states):
                transition_probs = (viterbi_table[t-1, :] * 
                                  self.transition_matrix[:, curr_state] * 
                                  self.emission_matrix[curr_state, obs_t])
                
                best_prev_state = np.argmax(transition_probs)
                viterbi_table[t, curr_state] = transition_probs[best_prev_state]
                path_table[t, curr_state] = best_prev_state
        
        best_path = np.zeros(T, dtype=int)
        
        best_path[T-1] = np.argmax(viterbi_table[T-1, :])
        best_prob = viterbi_table[T-1, best_path[T-1]]
        
        for t in range(T-2, -1, -1):
            best_path[t] = path_table[t+1, best_path[t+1]]
        
        return best_path, best_prob
    
    def decode_with_probabilities(self, probability_sequence):
        T, n_states = probability_sequence.shape
        
        if T == 0:
            return [], 0.0
        
        
        viterbi_table = np.zeros((T, self.n_states))
        path_table = np.zeros((T, self.n_states), dtype=int)
        
        viterbi_table[0, :] = self.initial_probs * probability_sequence[0, :]
        
        for t in range(1, T):
            for curr_state in range(self.n_states):
                transition_probs = (viterbi_table[t-1, :] * 
                                  self.transition_matrix[:, curr_state] * 
                                  probability_sequence[t, curr_state])
                
                best_prev_state = np.argmax(transition_probs)
                viterbi_table[t, curr_state] = transition_probs[best_prev_state]
                path_table[t, curr_state] = best_prev_state
        
        best_path = np.zeros(T, dtype=int)
        best_path[T-1] = np.argmax(viterbi_table[T-1, :])
        best_prob = viterbi_table[T-1, best_path[T-1]]
        
        for t in range(T-2, -1, -1):
            best_path[t] = path_table[t+1, best_path[t+1]]
        
        return best_path, best_prob
    
    def viterbi_decode_lightweight(self, obs_seq, use_log=True):
        if INFERENCE_OPTIMIZATION_CONFIG.get('use_fast_viterbi', False):
            return self.viterbi_decode_fast(obs_seq)
        
        S = self.n_states
        T = len(obs_seq)
        
        if T == 0:
            return np.array([], dtype=int), 0.0
        
        assert self.transition_matrix is not None, "Transition matrix not initialized"
        assert self.emission_matrix is not None, "Emission matrix not initialized"
        assert self.initial_probs is not None, "Initial probabilities not initialized"
        
        A = self.transition_matrix
        B = self.emission_matrix
        pi = self.initial_probs
        
        if use_log:
            logA = np.log(A + 1e-12)
            logB = np.log(B + 1e-12)
            logpi = np.log(pi + 1e-12)
            
            dp = np.zeros((T, S))
            path = np.zeros((T, S), dtype=int)
            
            dp[0] = logpi + logB[:, obs_seq[0]]
            
            for t in range(1, T):
                obs_t = obs_seq[t]
                for s in range(S):
                    probs = dp[t-1] + logA[:, s]
                    path[t, s] = np.argmax(probs)
                    dp[t, s] = probs[path[t, s]] + logB[s, obs_t]
            
            states = np.zeros(T, dtype=int)
            states[-1] = np.argmax(dp[-1])
            max_prob = dp[-1, states[-1]]
            
            for t in range(T-2, -1, -1):
                states[t] = path[t+1, states[t+1]]
            
            return states, np.exp(max_prob)
        
        else:
            return self.viterbi_decode(obs_seq)

    def viterbi_decode_fast(self, obs_seq):
        S = self.n_states
        T = len(obs_seq)
        
        if T == 0:
            return np.array([], dtype=np.int32), 0.0
        
        logA = np.log(self.transition_matrix + 1e-12)
        logB = np.log(self.emission_matrix + 1e-12)
        logpi = np.log(self.initial_probs + 1e-12)
        
        dtype = np.float32 if INFERENCE_OPTIMIZATION_CONFIG.get('use_float32', False) else np.float64
        dp = np.full((T, S), -np.inf, dtype=dtype)
        path = np.zeros((T, S), dtype=np.int32)
        
        dp[0] = logpi + logB[:, obs_seq[0]]
        
        for t in range(1, T):
            obs_t = obs_seq[t]
            for s in range(S):
                probs = dp[t-1] + logA[:, s]
                path[t, s] = np.argmax(probs)
                dp[t, s] = probs[path[t, s]] + logB[s, obs_t]
        
        states = np.zeros(T, dtype=np.int32)
        states[-1] = np.argmax(dp[-1])
        
        for t in range(T-2, -1, -1):
            states[t] = path[t+1, states[t+1]]
        
        return states, np.exp(dp[-1, states[-1]])

    def print_matrices(self):
        print("\n" + "="*60)
        print("HMM Model Matrices")
        print("="*60)
        
        print("\nTransition Probability Matrix (A):")
        print("Row: Previous State, Col: Current State")
        print("     ", end="")
        for j, name in enumerate(self.state_names):
            print(f"{name:>8}", end="")
        print()
        
        for i, name in enumerate(self.state_names):
            print(f"{name:>4}:", end="")
            for j in range(self.n_states):
                print(f"{self.transition_matrix[i, j]:>8.3f}", end="")
            print()
        
        if self.emission_matrix is not None:
            print("\nEmission Probability Matrix (B):")
            print("Row: True State, Col: Observed (Predicted)")
            print("     ", end="")
            for j, name in enumerate(self.state_names):
                print(f"{name:>8}", end="")
            print()
            
            for i, name in enumerate(self.state_names):
                print(f"{name:>4}:", end="")
                for j in range(self.n_states):
                    print(f"{self.emission_matrix[i, j]:>8.3f}", end="")
                print()
        
        print("\nInitial Probability Vector (π):")
        print("State: ", end="")
        for i, name in enumerate(self.state_names):
            print(f"{name:>8}", end="")
        print()
        print("Prob:  ", end="")
        for i, name in enumerate(self.state_names):
            print(f"{self.initial_probs[i]:>8.3f}", end="")
        print()
        print("="*60)

    def save_matrices_to_file(self, filepath, fold_idx=None):
        with open(filepath, 'a' if fold_idx is not None else 'w') as f:
            fold_str = f"FOLD {fold_idx} - " if fold_idx is not None else ""
            f.write(f"\n{'='*80}\n")
            f.write(f"{fold_str}HMM Model Matrices\n")
            f.write(f"{'='*80}\n")
            
            f.write("\nTransition Probability Matrix (A):\n")
            f.write("Row: Previous State, Col: Current State\n")
            f.write("     ")
            for j, name in enumerate(self.state_names):
                f.write(f"{name:>8}")
            f.write("\n")
            
            for i, name in enumerate(self.state_names):
                f.write(f"{name:>4}:")
                for j in range(self.n_states):
                    f.write(f"{self.transition_matrix[i, j]:>8.3f}")
                f.write("\n")
            
            if self.emission_matrix is not None:
                f.write("\nEmission Probability Matrix (B):\n")
                f.write("Row: True State, Col: Observed (Predicted)\n")
                f.write("     ")
                for j, name in enumerate(self.state_names):
                    f.write(f"{name:>8}")
                f.write("\n")
                
                for i, name in enumerate(self.state_names):
                    f.write(f"{name:>4}:")
                    for j in range(self.n_states):
                        f.write(f"{self.emission_matrix[i, j]:>8.3f}")
                    f.write("\n")
            
            f.write("\nInitial Probability Vector (π):\n")
            f.write("State: ")
            for i, name in enumerate(self.state_names):
                f.write(f"{name:>8}")
            f.write("\n")
            f.write("Prob:  ")
            for i, name in enumerate(self.state_names):
                f.write(f"{self.initial_probs[i]:>8.3f}")
            f.write("\n")
            f.write(f"{'='*80}\n")


def get_subject_sequences_from_fold(fold_idx, DataGenerator, label_type='true'):
    fold_len = DataGenerator.n // DataGenerator.k
    subject_sequences = []
    
    for p in range(DataGenerator.k):
        if p != fold_idx:
            for j in range(fold_len):
                subject_idx = p * fold_len + j
                if subject_idx >= len(DataGenerator.y_list):
                    break
                
                subject_labels = DataGenerator.y_list[subject_idx]
                
                if len(subject_labels.shape) > 1:
                    subject_labels = np.argmax(subject_labels, axis=1)
                
                subject_sequences.append(subject_labels)
    
    return subject_sequences


def train_hmm_model(train_true_labels, train_predicted_labels, fold_idx=None, DataGenerator=None, verbose=True, 
                   emission_max_samples=None, emission_sampling_strategy='stratified'):
    hmm = SleepStageHMM(n_states=5)
    
    if DataGenerator is not None and fold_idx is not None:
        train_sequences = get_subject_sequences_from_fold(fold_idx, DataGenerator, 'true')
        
        if verbose:
            print(f"Extracted {len(train_sequences)} subject sequences from training set")
            seq_lengths = [len(seq) for seq in train_sequences]
            print(f"Sequence length range: {min(seq_lengths)} - {max(seq_lengths)}")
            
            first_states = [seq[0] if len(seq) > 0 else -1 for seq in train_sequences]
            state_names = ['W', 'N1', 'N2', 'N3', 'REM']
            first_state_counts = np.bincount(first_states, minlength=5)
            print(f"First state distribution of subject sequences:")
            for j, (name, count) in enumerate(zip(state_names, first_state_counts)):
                print(f"  {name}: {count} subjects ({count/len(train_sequences)*100:.1f}%)")
    else:
        train_sequences = [train_true_labels]
        if verbose:
            print("Warning: DataGenerator not provided, using entire training set as a single sequence")
    
    hmm.estimate_transition_matrix_from_data(train_sequences)
    
    hmm.estimate_initial_probabilities_from_data(train_sequences)

    hmm.estimate_emission_matrix_from_confusion(
        train_true_labels, train_predicted_labels, 
        max_samples=emission_max_samples, 
        sampling_strategy=emission_sampling_strategy
    )
    
    if verbose:
        print(f"\nHMM model training completed (using only training data, {len(train_sequences)} sequences)")
        print(f"Training samples: {len(train_true_labels)}")
        if emission_max_samples is not None and emission_max_samples < len(train_true_labels):
            reduction_ratio = (1 - emission_max_samples / len(train_true_labels)) * 100
            print(f"Emission matrix sampling optimization: Reduced sample usage by {reduction_ratio:.1f}%")
        print(f"A matrix diagonal mean: {np.mean(np.diag(hmm.transition_matrix)):.3f}")
        print(f"B matrix diagonal mean: {np.mean(np.diag(hmm.emission_matrix)):.3f}")
        hmm.print_matrices()
    
    return hmm


def apply_hmm_smoothing(model_probabilities, hmm_model, fold_idx=None, verbose=True):
    
    assert hmm_model.transition_matrix is not None, "Transition matrix not correctly trained"
    assert hmm_model.emission_matrix is not None, "Emission matrix not correctly trained"
    assert hmm_model.initial_probs is not None, "Initial probabilities not correctly trained"
    
    if verbose:
        fold_str = f"Fold {fold_idx}: " if fold_idx is not None else ""
        print(f"\n{fold_str}Applying trained HMM model for smoothing")
        print("-" * 50)
        print(f"A matrix diagonal mean: {np.mean(np.diag(hmm_model.transition_matrix)):.3f}")
        print(f"B matrix diagonal mean: {np.mean(np.diag(hmm_model.emission_matrix)):.3f}")
    
    hard_predictions = np.argmax(model_probabilities, axis=1)
    hmm_predictions, hmm_path_prob = hmm_model.viterbi_decode_lightweight(hard_predictions)
    
    if verbose:
        print(f"\nHMM decoding completed:")
        print(f"Sequence length: {len(hmm_predictions)}")
        print(f"Path probability: {hmm_path_prob:.2e}")
        print(f"N3 self-transition probability: {hmm_model.transition_matrix[3,3]:.3f}")
        print(f"REM self-transition probability: {hmm_model.transition_matrix[4,4]:.3f}")
    
    return hmm_predictions, hmm_path_prob

INFERENCE_OPTIMIZATION_CONFIG = {
    'batch_size_multiplier': 4,
    'use_fast_savgol': True,
    'use_fast_viterbi': True,
    'enable_model_cache': True,
    'reduce_verbose': True,
    'use_float32': True,
    'enable_memory_cleanup': True
}

def predict_with_optimized_batch(ce_model, data, original_batch_size=32, multiplier=4):
    optimized_batch_size = min(original_batch_size * multiplier, len(data))
    
    predictions = ce_model.predict(data, batch_size=optimized_batch_size, verbose=0)
    return predictions

def cleanup_fold_resources():
    keras.backend.clear_session()
    tf.keras.backend.clear_session()
    gc.collect()
    
    if tf.config.list_physical_devices('GPU'):
        try:
            tf.config.experimental.reset_memory_stats('GPU:0')
        except:
            pass

MODEL_CACHE = {}
HMM_MODEL_CACHE = {}

def save_hmm_model(hmm_model, filepath):
    np.savez_compressed(filepath, 
                       transition_matrix=hmm_model.transition_matrix,
                       emission_matrix=hmm_model.emission_matrix,
                       initial_probs=hmm_model.initial_probs)

def load_hmm_model(filepath):
    data = np.load(filepath)
    hmm = SleepStageHMM()
    hmm.transition_matrix = data['transition_matrix']
    hmm.emission_matrix = data['emission_matrix'] 
    hmm.initial_probs = data['initial_probs']
    return hmm

def get_or_load_model(fold_idx, model_path, input_shape, freq, channels, time_second):
    if not INFERENCE_OPTIMIZATION_CONFIG.get('enable_model_cache', False):
        model = create_model_light(input_shape=input_shape, 
                                 freq=freq, channels=channels, time_second=time_second)
        model.load_weights(model_path, skip_mismatch=True, by_name=True)
        return model
        
    if fold_idx not in MODEL_CACHE:
        model = create_model_light(input_shape=input_shape, 
                                 freq=freq, channels=channels, time_second=time_second)
        model.load_weights(model_path, skip_mismatch=True, by_name=True)
        MODEL_CACHE[fold_idx] = model
    return MODEL_CACHE[fold_idx]

def apply_hmm_smoothing_fast(model_probabilities, hmm_model, fold_idx=None, verbose=False):
    assert hmm_model.transition_matrix is not None, "Transition matrix not correctly trained"
    assert hmm_model.emission_matrix is not None, "Emission matrix not correctly trained"
    assert hmm_model.initial_probs is not None, "Initial probabilities not correctly trained"
    
    hard_predictions = np.argmax(model_probabilities, axis=1)
    
    if INFERENCE_OPTIMIZATION_CONFIG.get('use_fast_viterbi', False):
        hmm_predictions, hmm_path_prob = hmm_model.viterbi_decode_fast(hard_predictions)
    else:
        hmm_predictions, hmm_path_prob = hmm_model.viterbi_decode_lightweight(hard_predictions)
    
    if verbose:
        print(f"Fold {fold_idx}: HMM decoding completed, sequence length: {len(hmm_predictions)}")
    
    return hmm_predictions, hmm_path_prob

import gc

if INFERENCE_OPTIMIZATION_CONFIG.get('reduce_verbose', True):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# k-fold cross validation
all_scores = []

hmm_matrices_file = output_path + "HMM_matrices.txt"
if os.path.exists(hmm_matrices_file):
    os.remove(hmm_matrices_file)
print(f"HMM matrices will be saved to: {hmm_matrices_file}")


first_decay_steps = 10
lr_decayed_fn = (
  tf.keras.optimizers.schedules.CosineDecayRestarts(
      0.001,
      first_decay_steps))


tf.config.experimental_run_functions_eagerly(True)


best_val_acc = []
all_scores = []
all_scores_smooth = []
all_scores_smooth_hmm = []
all_hmm_models = []

for i in range(0, fold):  # 20-fold
    print('Fold #', i)

    train_data, train_targets, val_data, val_targets = DataGenerator.getFold(i)  # train_data [7665, 10, 3000] val_data[924, 10, 3000] train_targets[7665, 5] val_targets[924, 5]
    train_data, val_data = train_data.reshape(-1, 30 * freq, channels), val_data.reshape(-1, 30 * freq, channels)  # train_data [7665, 3000, 10] val_data[924, 3000, 10]
    

    opt = tf.keras.optimizers.Adam(learning_rate=lr_decayed_fn, amsgrad=True)
    ce_model = create_model_light(input_shape=train_data.shape[1:], freq=freq, channels=channels, time_second=30)
    verbose = 1

    ce_model.compile(
        optimizer=opt,
        loss={'Label': "categorical_crossentropy"},
        metrics={'Label': "accuracy"}
    )
    
    if not os.path.exists(output_path+str(i)+'ResNet_Best'+'.h5'):

        history = ce_model.fit(
            train_data, train_targets,
            batch_size=cfg['bs'], epochs=cfg['epochs'],
            validation_data=(val_data, val_targets),
            callbacks=[
                tf.keras.callbacks.ModelCheckpoint(
                    output_path + str(i) + 'ResNet_Best' + '.h5',
                    monitor='val_accuracy',
                    verbose=1,
                    save_best_only=True,
                    mode='auto')],
            verbose=verbose
        )

        # Save training information
        fit_loss = np.array(history.history['loss'])
        fit_acc = np.array(history.history['accuracy'])
        fit_val_loss = np.array(history.history['val_loss'])
        fit_val_acc = np.array(history.history['val_accuracy'])
        print('Best val acc:', max(history.history['val_accuracy']))
        best_val_acc.append(max(history.history['val_accuracy']))

        saveFile = open(output_path + "Result_MFE.txt", 'a+')
        print('Fold #'+str(i), file=saveFile)
        print(history.history, file=saveFile)
        saveFile.close()    


    # get and save the learned feature
    ce_model.load_weights(output_path+str(i)+'ResNet_Best'+'.h5', skip_mismatch=True, by_name=True)

   
    batch_multiplier = INFERENCE_OPTIMIZATION_CONFIG.get('batch_size_multiplier', 1)
    predictions = predict_with_optimized_batch(ce_model, val_data, cfg['bs'], batch_multiplier)

    AllPred_temp_original = np.argmax(predictions, axis=1)
    AllTrue_temp = np.argmax(val_targets, axis=1)

    acc = metrics.accuracy_score(AllTrue_temp, AllPred_temp_original)
    print(f"Fold {i} - Original Accuracy: {acc:.4f}")
    all_scores.append(acc)
    
    print(f"Fold {i}: Applying Savitzky-Golay probability smoothing")
    
    window_length = 5  
    polyorder = 2      
    
    smoothed_probs, AllPred_temp = savgol_smooth_probabilities(
        predictions, 
        window_length=window_length, 
        polyorder=polyorder
    )

    acc_smooth = metrics.accuracy_score(AllTrue_temp, AllPred_temp)
    print(f"Fold {i} - Savitzky-Golay Accuracy: {acc_smooth:.4f}")
    all_scores_smooth.append(acc_smooth)
    
    print(f"Fold {i}: Training HMM model using training set")
    
    train_predictions = predict_with_optimized_batch(ce_model, train_data, cfg['bs'], batch_multiplier)
    
    train_smoothed_probs, train_pred_labels = savgol_smooth_probabilities(
        train_predictions, 
        window_length=window_length, 
        polyorder=polyorder
    )
    train_true_labels = np.argmax(train_targets, axis=1)
    
    HMM_VERBOSE = not INFERENCE_OPTIMIZATION_CONFIG.get('reduce_verbose', True)
    
    try:
        hmm_cache_file = output_path + f"hmm_model_fold_{i}.npz"
        
        if INFERENCE_OPTIMIZATION_CONFIG.get('enable_model_cache', True) and os.path.exists(hmm_cache_file):
            if not HMM_VERBOSE:
                print(f"Fold {i}: Loading cached HMM model")
            hmm_model = load_hmm_model(hmm_cache_file)
        else:
            emission_max_samples = HMM_EMISSION_CONFIG['max_samples'] if HMM_EMISSION_CONFIG['enable_sampling'] else None
            emission_sampling_strategy = HMM_EMISSION_CONFIG['sampling_strategy']
            
            hmm_model = train_hmm_model(
                train_true_labels,
                train_pred_labels,
                fold_idx=i,
                DataGenerator=DataGenerator,
                verbose=HMM_VERBOSE,
                emission_max_samples=emission_max_samples,
                emission_sampling_strategy=emission_sampling_strategy
            )
            
            if INFERENCE_OPTIMIZATION_CONFIG.get('enable_model_cache', True):
                save_hmm_model(hmm_model, hmm_cache_file)
        
        if HMM_VERBOSE:
            print(f"\n{'='*80}")
            print(f"FOLD {i} - HMM Model Matrices After Training")
            print(f"{'='*80}")
            hmm_model.print_matrices()
            print(f"{'='*80}")
        
        hmm_model.save_matrices_to_file(output_path + "HMM_matrices.txt", fold_idx=i)
        
        all_hmm_models.append(hmm_model)
        
        if INFERENCE_OPTIMIZATION_CONFIG.get('reduce_verbose', True):
            AllPred_temp_hmm, hmm_path_prob = apply_hmm_smoothing_fast(
                smoothed_probs,
                hmm_model,
                fold_idx=i, 
                verbose=HMM_VERBOSE
            )
        else:
            AllPred_temp_hmm, hmm_path_prob = apply_hmm_smoothing(
                smoothed_probs,
                hmm_model,
                fold_idx=i, 
                verbose=HMM_VERBOSE
            )
        
        acc_hmm = metrics.accuracy_score(AllTrue_temp, AllPred_temp_hmm)
        print(f"Fold {i} - No Data Leakage Savitzky-Golay + HMM Accuracy: {acc_hmm:.4f}")
        all_scores_smooth_hmm.append(acc_hmm)
        
        AllPred_temp_final = AllPred_temp_hmm
    except Exception as e:
        print(f"  ❌ HMM processing failed: {e}")
        print(f"  Using Savitzky-Golay smoothing result as final prediction")
        AllPred_temp_final = AllPred_temp
        all_scores_smooth_hmm.append(acc_smooth)
    
    
    AllPred_temp_hmm_final = AllPred_temp_final
    
    AllPred_temp = AllPred_temp_final

    if i == 0:
        AllPred = AllPred_temp
        AllTrue = AllTrue_temp
        AllPred_HMM = AllPred_temp_hmm_final
        AllTrue_HMM = AllTrue_temp
    else:
        AllPred = np.concatenate((AllPred, AllPred_temp))
        AllTrue = np.concatenate((AllTrue, AllTrue_temp))
        AllPred_HMM = np.concatenate((AllPred_HMM, AllPred_temp_hmm_final))
        AllTrue_HMM = np.concatenate((AllTrue_HMM, AllTrue_temp))

    # VariationCurve(fit_acc, fit_val_acc, f'Acc_{i}', output_path, figsize=(9, 6))
    # VariationCurve(fit_loss, fit_val_loss, f'Loss_{i}', output_path, figsize=(9, 6))

    print(128*'=')
    print(f"Fold {i} - Standard Evaluation Results (No Data Leakage Savitzky-Golay + HMM):")
    PrintScore(AllTrue_temp, AllPred_temp, fold=i, savePath=output_path)
    ConfusionMatrix(AllTrue_temp, AllPred_temp, fold=i, classes=['W', 'N1', 'N2', 'N3', 'REM'], savePath=output_path)
    
    print(f"\nFold {i} - Separate Evaluation of No Data Leakage HMM Optimized Results:")
    PrintScore(AllTrue_temp, AllPred_temp_hmm_final, fold=f"{i}_HMM_NoLeak", savePath=output_path)
    ConfusionMatrix(AllTrue_temp, AllPred_temp_hmm_final, fold=f"{i}_HMM_NoLeak", classes=['W', 'N1', 'N2', 'N3', 'REM'], savePath=output_path)

    if INFERENCE_OPTIMIZATION_CONFIG.get('enable_memory_cleanup', True):
        cleanup_fold_resources()
    else:
        keras.backend.clear_session()
        gc.collect()
    
    del train_data, train_targets, val_data, val_targets
    print('Fold #', i, 'finished')



print(128 * '_')
print('End of training MFE.')
print(128 * '#')

# print acc of each fold
if len(best_val_acc) != 0:
    print(128*'=')
    print("best val acc: ",best_val_acc)
    print("Average best val acc of each fold: ",np.mean(best_val_acc))

print(128*'=')
print("Original acc: ",all_scores)
print("Average original acc: ",np.mean(all_scores))

print(128*'=')
print("Savitzky-Golay smoothed acc: ",all_scores_smooth)
print("Average smoothed acc: ",np.mean(all_scores_smooth))

print(128*'=')
print("No Data Leakage Savitzky-Golay + HMM acc: ",all_scores_smooth_hmm)
print("Average No Data Leakage Savitzky-Golay + HMM acc: ",np.mean(all_scores_smooth_hmm))
print("Average acc of each fold: ",np.mean(all_scores_smooth_hmm))

if len(all_hmm_models) > 0:
    print(f"\n{'='*80}")
    print(f"Average HMM Matrices of all {len(all_hmm_models)} FOLDS")
    print(f"{'='*80}")
    
    avg_transition = np.mean([model.transition_matrix for model in all_hmm_models], axis=0)
    
    avg_emission = np.mean([model.emission_matrix for model in all_hmm_models], axis=0)
    
    avg_initial = np.mean([model.initial_probs for model in all_hmm_models], axis=0)
    
    state_names = ['W', 'N1', 'N2', 'N3', 'REM']
    
    print("\nAverage Transition Probability Matrix (A_avg):")
    print("Row: Previous State, Col: Current State")
    print("     ", end="")
    for name in state_names:
        print(f"{name:>8}", end="")
    print()
    
    for i, name in enumerate(state_names):
        print(f"{name:>4}:", end="")
        for j in range(5):
            print(f"{avg_transition[i, j]:>8.3f}", end="")
        print()
    
    print("\nAverage Emission Probability Matrix (B_avg):")
    print("Row: True State, Col: Observed (Predicted)")
    print("     ", end="")
    for name in state_names:
        print(f"{name:>8}", end="")
    print()
    
    for i, name in enumerate(state_names):
        print(f"{name:>4}:", end="")
        for j in range(5):
            print(f"{avg_emission[i, j]:>8.3f}", end="")
        print()
    
    print("\nAverage Initial Probability Vector (π_avg):")
    print("State: ", end="")
    for name in state_names:
        print(f"{name:>8}", end="")
    print()
    print("Prob:  ", end="")
    for i, prob in enumerate(avg_initial):
        print(f"{prob:>8.3f}", end="")
    print()
    print(f"{'='*80}")
    
    with open(output_path + "HMM_matrices.txt", 'a') as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"Average HMM Matrices of all {len(all_hmm_models)} FOLDS\n")
        f.write(f"{'='*80}\n")
        
        f.write("\nAverage Transition Probability Matrix (A_avg):\n")
        f.write("Row: Previous State, Col: Current State\n")
        f.write("     ")
        for name in state_names:
            f.write(f"{name:>8}")
        f.write("\n")
        
        for i, name in enumerate(state_names):
            f.write(f"{name:>4}:")
            for j in range(5):
                f.write(f"{avg_transition[i, j]:>8.3f}")
            f.write("\n")
        
        f.write("\nAverage Emission Probability Matrix (B_avg):\n")
        f.write("Row: True State, Col: Observed (Predicted)\n")
        f.write("     ")
        for name in state_names:
            f.write(f"{name:>8}")
        f.write("\n")
        
        for i, name in enumerate(state_names):
            f.write(f"{name:>4}:")
            for j in range(5):
                f.write(f"{avg_emission[i, j]:>8.3f}")
            f.write("\n")
        
        f.write("\nAverage Initial Probability Vector (π_avg):\n")
        f.write("State: ")
        for name in state_names:
            f.write(f"{name:>8}")
        f.write("\n")
        f.write("Prob:  ")
        for i, prob in enumerate(avg_initial):
            f.write(f"{prob:>8.3f}")
        f.write("\n")
        f.write(f"{'='*80}\n")



# Print score to console
print(128*'=')
print("Global Evaluation Results - Final Optimized Results (No Data Leakage Savitzky-Golay + HMM):")
PrintScore(AllTrue, AllPred, savePath=output_path)

# Print confusion matrix and save
ConfusionMatrix(AllTrue, AllPred, classes=['W','N1','N2','N3','REM'], savePath=output_path)

print(128*'=')
print("Global Evaluation Results - Separate Evaluation of No Data Leakage HMM Optimized Results:")
PrintScore(AllTrue_HMM, AllPred_HMM, fold="Global_HMM_NoLeak", savePath=output_path)
ConfusionMatrix(AllTrue_HMM, AllPred_HMM, fold="Global_HMM_NoLeak", classes=['W','N1','N2','N3','REM'], savePath=output_path)

print('End of evaluating MFE.')
print('###train without contrastive learning###')
print(128 * '#')

del Fold_Data
gc.collect()


def quick_test_emission_sampling():
    print("="*60)
    print("HMM Emission Matrix Sampling Optimization Test")
    print("="*60)
    
    np.random.seed(42)
    n_samples = 20000
    true_labels = np.random.choice(5, n_samples, p=[0.3, 0.15, 0.25, 0.2, 0.1])
    noise_rate = 0.1
    predicted_labels = true_labels.copy()
    noise_mask = np.random.random(n_samples) < noise_rate
    predicted_labels[noise_mask] = np.random.choice(5, np.sum(noise_mask))
    
    hmm = SleepStageHMM(n_states=5)
    
    configs = [
        (None, 'stratified', 'All Samples'),
        (5000, 'stratified', '5K Stratified'),
        (5000, 'balanced', '5K Balanced'),
        (5000, 'random', '5K Random'),
        (2000, 'stratified', '2K Stratified'),
    ]
    
    results = []
    for max_samples, strategy, desc in configs:
        import time
        start_time = time.time()
        
        B = hmm.estimate_emission_matrix_from_confusion(
            true_labels, predicted_labels, 
            max_samples=max_samples, 
            sampling_strategy=strategy
        )
        
        elapsed = time.time() - start_time
        diagonal_mean = np.mean(np.diag(B))
        
        results.append((desc, elapsed, diagonal_mean))
        print(f"{desc:15s} - Time: {elapsed:.3f}s, Diagonal Mean: {diagonal_mean:.3f}")
    
    print("="*60)
    return results

# quick_test_emission_sampling()

# ================================================================================
# ================================================================================

def print_optimization_summary():
    print("\n" + "="*80)
    print("Inference Performance Optimization Configuration Summary")
    print("="*80)
    
    config = INFERENCE_OPTIMIZATION_CONFIG
    print(f"Batch Size Multiplier: {config.get('batch_size_multiplier', 1)}x (Original: {cfg['bs']} -> {cfg['bs'] * config.get('batch_size_multiplier', 1)})")
    print(f"Fast Savitzky-Golay: {'✓' if config.get('use_fast_savgol', False) else '✗'}")
    print(f"Fast Viterbi Algorithm: {'✓' if config.get('use_fast_viterbi', False) else '✗'}")
    print(f"Model Cache: {'✓' if config.get('enable_model_cache', False) else '✗'}")
    print(f"Reduce Verbose Output: {'✓' if config.get('reduce_verbose', False) else '✗'}")
    print(f"Use Float32 Precision: {'✓' if config.get('use_float32', False) else '✗'}")
    print(f"Memory Cleanup Optimization: {'✓' if config.get('enable_memory_cleanup', False) else '✗'}")
    
    hmm_config = HMM_EMISSION_CONFIG
    print(f"\nHMM Emission Matrix Sampling Optimization:")
    print(f"  Enable Sampling: {'✓' if hmm_config.get('enable_sampling', False) else '✗'}")
    if hmm_config.get('enable_sampling', False):
        print(f"  Max Samples: {hmm_config.get('max_samples', 'N/A')}")
        print(f"  Sampling Strategy: {hmm_config.get('sampling_strategy', 'N/A')}")
    
    print("="*80)
    print("Performance Optimization Suggestions:")
    print("1. First run recommendation: batch_size_multiplier=4, reduce_verbose=True")
    print("2. Fast test mode: Enable all optimization options")
    print("3. Production environment: Enable cache and sampling optimization, disable verbose output")
    print("4. Large datasets: Set HMM sampling max_samples=5000-10000")
    print("="*80)

def benchmark_optimization_impact():
    print("\n" + "="*80)
    print("Performance Optimization Impact Benchmark")
    print("="*80)
    
    import time
    
    np.random.seed(42)
    n_samples = 1000
    n_classes = 5
    mock_probs = np.random.rand(n_samples, n_classes)
    mock_probs = mock_probs / mock_probs.sum(axis=1, keepdims=True)
    
    print("Savitzky-Golay Filtering Performance Comparison:")
    
    start = time.time()
    _, _ = savgol_smooth_probabilities(mock_probs)
    original_time = time.time() - start
    
    start = time.time()
    _, _ = savgol_smooth_probabilities_fast(mock_probs)
    fast_time = time.time() - start
    
    print(f"  Original Implementation: {original_time:.4f}s")
    print(f"  Fast Implementation: {fast_time:.4f}s")
    print(f"  Performance Improvement: {original_time/fast_time:.2f}x")
    
    print("\nHMM Emission Matrix Sampling Performance Comparison:")
    
    large_n = 50000
    true_labels = np.random.choice(5, large_n)
    pred_labels = np.random.choice(5, large_n)
    
    hmm = SleepStageHMM()
    
    start = time.time()
    hmm.estimate_emission_matrix_from_confusion(true_labels, pred_labels, max_samples=None)
    full_time = time.time() - start
    
    start = time.time()
    hmm.estimate_emission_matrix_from_confusion(true_labels, pred_labels, 
                                               max_samples=5000, sampling_strategy='stratified')
    sampled_time = time.time() - start
    
    print(f"  Full Samples ({large_n}): {full_time:.4f}s")
    print(f"  Sampled Version (5000): {sampled_time:.4f}s")
    print(f"  Performance Improvement: {full_time/sampled_time:.2f}x")
    print(f"  Sample Reduction: {(1-5000/large_n)*100:.1f}%")
    
    print("="*80)

print_optimization_summary()