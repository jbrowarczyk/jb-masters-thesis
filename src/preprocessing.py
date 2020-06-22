from global_settings import EEG_CHANNELS, N_CHANNELS, N_SAMPLES
from sklearn.decomposition import FastICA
from sklearn.exceptions import ConvergenceWarning
import warnings
import numpy as np
import scipy.signal
import scipy.stats
import spectrum
import matplotlib.pyplot as plt
import pywt
import random
import sys

#filter scikit-learn ICA convergence warnings
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

# doesn't increase classification accuracy
def compute_additional_stats(eeg,indices):
    output_structure = {}
    for subset_name in indices.keys():
        output_structure.update({subset_name : {}})
        for class_name in indices[subset_name].keys():
            output_structure[subset_name].update({class_name : []})
            for index in indices[subset_name][class_name]:
                stats = np.ndarray(shape=(6,eeg[class_name][index].shape[1]),dtype='float64')
                for i in range(eeg[class_name][index].shape[1]):
                    s = np.ndarray(shape=(6,),dtype='float64')
                    channel = np.array(eeg[class_name][index][:,i]).reshape((128,))
                    s[0] = np.mean(channel)
                    s[1] = np.var(channel)
                    s[2] = scipy.stats.skew(channel)
                    s[3] = scipy.stats.kurtosis(channel)
                    s[4] = zero_crossings(channel)
                    s[5] = sum(np.power(channel,2))
                    stats[:,i] = s
                output_structure[subset_name][class_name].append(stats)
    return output_structure

def compute_stats(eeg,indices):
    output_structure = {}
    for subset_name in indices.keys():
        output_structure.update({subset_name : {}})
        for class_name in indices[subset_name].keys():
            output_structure[subset_name].update({class_name : []})
            for index in indices[subset_name][class_name]:
                stats = np.ndarray(shape=(2,eeg[class_name][index].shape[1]),dtype='float64')
                for i in range(eeg[class_name][index].shape[1]):
                    s = np.ndarray(shape=(2,),dtype='float64')
                    s[0] = np.mean(eeg[class_name][index][:,i])
                    s[1] = np.var(eeg[class_name][index][:,i])
                    stats[:,i] = s
                output_structure[subset_name][class_name].append(stats)
    return output_structure

def concat_data_into_lists_of_frames(eeg):
    output_structure = {}
    for class_name in eeg.keys():
        output_structure.update({class_name : []})
        for item1 in eeg[class_name]:
            for item2 in item1:
                output_structure[class_name].append(item2)
    return output_structure

def divide_into_train_val_test(eeg,train=0.7,val=0.1,test=0.2):
    if 1.0 - (np.float64(train) + np.float64(val) + np.float64(test)) > np.finfo(np.float64).eps:
        raise ValueError("Sum of sample proportions not equal to 100")
    output_structure = {}
    indices_structure = {}
    output_structure.update({'train' : {}})
    output_structure.update({'val' : {}})
    output_structure.update({'test' : {}})
    indices_structure.update({'train' : {}})
    indices_structure.update({'val' : {}})
    indices_structure.update({'test' : {}})
    for class_name in eeg.keys():
        output_structure['train'].update({class_name : None})
        output_structure['val'].update({class_name : None})
        output_structure['test'].update({class_name : None})
        indices_structure['train'].update({class_name : None})
        indices_structure['val'].update({class_name :  None})
        indices_structure['test'].update({class_name : None})
        indices = [i for i in range(len(eeg[class_name]))]
        random.shuffle(indices)
        a = int(train*len(indices))
        b = a + int(val*len(indices))
        data_train = [ eeg[class_name][i] for i in indices[:a] ]
        data_val = [ eeg[class_name][i] for i in indices[a:b] ]
        data_test = [ eeg[class_name][i] for i in indices[b:] ]
        output_structure['train'][class_name]= data_train
        output_structure['val'][class_name] = data_val
        output_structure['test'][class_name] = data_test
        indices_structure['train'][class_name] = indices[:a]
        indices_structure['val'][class_name] = indices[a:b]
        indices_structure['test'][class_name] = indices[b:]
    return output_structure, indices_structure

def detrend_data(eeg):
    output_structure = {}
    for subset_name in eeg.keys():
        output_structure.update({subset_name : {}})
        for class_name in eeg[subset_name].keys():
            output_structure[subset_name].update({class_name : []})
            for i in range(len(eeg[subset_name][class_name])):
                dataframe = np.matrix(eeg[subset_name][class_name][i], copy=True)
                for j in range(len(EEG_CHANNELS)):
                    dataframe[:,j] = scipy.signal.detrend(dataframe[:,j], axis=0)
                output_structure[subset_name][class_name].append(dataframe)
    return output_structure

def perform_ica(eeg,whiten=True):
    ica = FastICA(n_components=14, whiten=whiten)
    output_structure = {}
    for subset_name in eeg.keys():
        output_structure.update({subset_name : {}})
        for class_name in eeg[subset_name].keys():
            output_structure[subset_name].update({class_name : []})
            for i in range(len(eeg[subset_name][class_name])):
                    dataframe = eeg[subset_name][class_name][i]
                    output_structure[subset_name][class_name].append(ica.fit_transform(dataframe))
    return output_structure

def reshape_data(eeg):
    output_structure = {}
    for class_name in eeg.keys():
        output_structure.update({class_name : []})
        for i in range(len(eeg[class_name])):
            output_structure[class_name].append([])
            for j in range(len(eeg[class_name][i])):
                if eeg[class_name][i][j].shape != (N_SAMPLES,N_CHANNELS) and eeg[class_name][i][j].shape != (N_CHANNELS,N_SAMPLES):
                    raise TypeError("Wrong data shape")
                elif eeg[class_name][i][j].shape == (N_CHANNELS,N_SAMPLES):
                    output_structure[class_name][i].append(eeg[class_name][i][j].transpose())
                elif eeg[class_name][i][j].shape == (N_SAMPLES,N_CHANNELS):
                    output_structure[class_name][i].append(eeg[class_name][i][j])
    return output_structure


def main():
    try:
        print("Loading raw data...")
        raw_data = np.load('eeg_signals.npy',allow_pickle=True).item()
        print("Raw data loaded")

    except Exception as e:
        print("Error during loading raw data")
        print(e)
        sys.exit()

    try:
        print("Preprocessing raw data...")
        transposed_data = reshape_data(raw_data)
        list_of_frames = concat_data_into_lists_of_frames(transposed_data)

        print("Dividing data into training, validation and test data...")
        divided_data, indices = divide_into_train_val_test(list_of_frames,train=0.7,val=0.1,test=0.2)

        print("Computing means and variances from raw data...")
        stats = compute_stats(list_of_frames,indices)

        print("Detrending data...")
        detrended_data = detrend_data(divided_data)

        print("Performing ICA...")
        ica_data = perform_ica(detrended_data)

        print("Data preprocessed")

    except Exception as e:
        print("Error during preprocessing raw data")
        print(e)
        sys.exit()

    try:
        print("Saving preprocessed data...")
        np.save("preprocessed_data.npy",ica_data)
        np.save("indices.npy",indices)
        np.save("stats.npy",stats)
        print("Preprocessed data saved")

    except Exception as e:
        print("Error during saving preprocessing data")
        print(e)
        sys.exit()

if __name__ == "__main__":
    main()

