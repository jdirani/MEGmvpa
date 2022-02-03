import mne
import numpy as np
from sklearn.decomposition import PCA
from mne.decoding import UnsupervisedSpatialFilter


def sliding_average(epochs, decim=5):
    '''
    Sliding average time window on the epochs data.
    The time windows are non overlapping. For each time slice, the average of all
    data at those time points is kept (as opposed to keeping all time points as a
    vector).
    For overlapping time windows that keep all time points as a vector, see
    make_temporal_patches()

    epochs : mne.epochs.EpochsFIF or np.array of epochs/stc data
            if np.array, must have dimensions (ntrials x nsource/sensor x ntimes)

    decim : integer to decimate by

    returns
    --------
    downsampled_epoch : np.array of the downsampled epochs

    '''
    if type(epochs) == mne.epochs.EpochsFIF: # convert to np.array if type is mne epochs obj
        epochs_data = epochs.get_data()
    else:
        epochs_data = epochs

    n_times =  epochs_data.shape[2]
    trimmed_times = n_times - n_times%decim # remove extra time points so n_times is divisible by decim
    epochs_data = epochs_data[:,:,:trimmed_times] # drop the additional time points at the end

    target_nTimes = int(trimmed_times / decim) # nb of times for the decimated epochs
    downsampled_epoch = np.zeros([epochs_data.shape[0],epochs_data.shape[1], target_nTimes]) # create output array with zeros
    for i_trial in range(epochs_data.shape[0]): # for each trial
        for i_sensor in range(epochs_data.shape[1]): # for each sensor

            sub = epochs_data[i_trial, i_sensor,:]
            downsampled_epoch[i_trial, i_sensor,:] = np.mean(sub.reshape(-1, decim), axis=1)

    return downsampled_epoch


def PCA_epochs(epochs, n_comp):
    '''
    Fit PCA on epochs_fit, and transform epochs_trans using n_comp.
    Prints explained variance when using n_comp.
    '''
    mdl_PCA = PCA(n_comp)
    pca = UnsupervisedSpatialFilter(mdl_PCA, average=False)
    pca_data = pca.fit_transform(epochs.get_data())

    explained_var = np.cumsum(mdl_PCA.explained_variance_ratio_)[-1]
    print('PCA explained var:%.3f'%explained_var)
    return pca_data



def PCA_epochsArray(array, n_comp):
    '''
    Fit PCA on epochs_fit, and transform epochs_trans using n_comp.
    Prints explained variance when using n_comp.
    '''
    mdl_PCA = PCA(n_comp)
    pca = UnsupervisedSpatialFilter(mdl_PCA, average=False)
    pca_data = pca.fit_transform(array)

    explained_var = np.cumsum(mdl_PCA.explained_variance_ratio_)[-1]
    print('PCA explained var:%.3f'%explained_var)
    return pca_data


def get_label_data(stcs_epochs, subj, label_name, parc, src_path, subjects_dir):
    '''
    returns Source Timecourse data in a numpy array with dimensions n_trail x n_sources x n_times

    stcs_epochs : list
                  list of stc objects
    subj : str
           subject name e.g. 'fsaverage'
    label_name : str
                 e.g. 'Brodmann.18-lh'
    parc :  str
            e.g. 'PALS_B12_Brodmann'
    src_path : str
               sourcespace file path
    '''

    src = mne.read_source_spaces(fname=src_path)
    parc = mne.read_labels_from_annot(subj, parc=parc ,subjects_dir=subjects_dir)
    label = [i for i in parc if i.name==label_name][0]
    label_data = np.array([stc.in_label(label).data for stc in stcs_epochs])
    return label_data
