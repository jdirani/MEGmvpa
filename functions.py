import mne, scipy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from os.path import join
from warnings import warn

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import RidgeClassifierCV, LogisticRegressionCV
from sklearn.decomposition import PCA
from sklearn.utils import class_weight
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score

from mne.decoding import GeneralizingEstimator, SlidingEstimator, cross_val_multiscore, UnsupervisedSpatialFilter, Vectorizer
from mne.decoding import Scaler, get_coef, LinearModel
from mne.stats import linear_regression, fdr_correction

from scipy.stats import pearsonr
from scipy.spatial import distance
from itertools import combinations
from pingouin import partial_corr
from numpy import arctanh
from tqdm import tqdm


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


def get_label_data(stcs_epochs, subj, label_name, parc, src_path, subjects_dir):
    '''
    returns stc data in a numpy array with dimensions n_trail x n_sources x n_times

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


# ================================= PCA ====================================== #
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

# def PCA_evokeds_ideogram(evokeds_image, evokeds_word, n_comp):
#     '''
#     Fit PCA on grand average of all evokeds, then transform evokeds_image and evokeds_word.
#
#     evokeds_image : np.array of evoked responses for the image modality (ntrials x nsensors x ntimes)
#     evokeds_words : np.array of evoked responses for the image modality (ntrials x nsensors x ntimes)
#     n_comp : number of components to keep
#
#     '''
#
#     all_evokeds_data = np.array([i.data for i in evokeds_image + evokeds_word])  # concat data of all evokeds
#     mdl_PCA = PCA(n_comp)
#     pca = UnsupervisedSpatialFilter(mdl_PCA, average=False)
#     pca.fit(X=all_evokeds_data) # fit on the concatenated evokeds
#     # transform seperately on images and words
#     pca_evokeds_image = pca.transform(np.array([i.data for i in evokeds_image]))
#     pca_evokeds_word = pca.transform(np.array([i.data for i in evokeds_word]))
#
#     explained_var = np.cumsum(mdl_PCA.explained_variance_ratio_)[-1]
#     print('PCA explained var:%.3f'%explained_var)
#
#     return pca_evokeds_image, pca_evokeds_word



# ============================= DECODING ===================================== #

def temporal_decoding(X, y, mdl, scoring, cv=5, n_jobs=1):
    '''
    Temporal decoding using sliding estimator.
    Train/fit at each time point of X.
    Scores calculated using cross_val_multiscore: use simple models (e.g LogisticRegression
    as opposed to LogisticRegressionCV).

    X : np.array wit shape (n_trials, n_sensors, n_times)
    y : np.array
    mdl : sklearn model
    scoring: str, scoring method from sklearn

    returns
    scores : np.array, score at each time point.

    '''

    clf = make_pipeline(StandardScaler(),  mdl) # create the model and pipeline
    time_decod = SlidingEstimator(clf, n_jobs=n_jobs, scoring=scoring, verbose=True)
    scores = cross_val_multiscore(time_decod, X, y, cv=cv, n_jobs=n_jobs)
    scores = np.mean(scores, axis=0) # Mean scores across cross-validation splits

    return scores


def temporal_decoding_slidingwindow():
    '''
    Temporal decoding using sliding estimator.
    Train/fit using sliding time window of N points around each time point of X.
    The mne SlidingEstimator is not used and so we need to manually cross validate.
    '''

    pass

def plot_temporal_scores(scores, times, fname, chance_level=None, figsize=[12,4], title='Temporal decoding'):
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(times, scores, label='scores')
    if chance_level:
        ax.axhline(chance_level, color='k', linestyle='--', label='chance')

    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Score')
    ax.legend()
    ax.set_title(title)
    if fname:
        fig.savefig(fname)
        plt.close()


# Not working. See : https://mne.tools/dev/auto_tutorials/machine-learning/50_decoding.html#sphx-glr-auto-tutorials-machine-learning-50-decoding-py
# def plot_temporal_decoding_topomap(X, y, mdl, scoring, cv=5, times, info, n_jobs=1):
#
#     clf = make_pipeline(StandardScaler(), LinearModel(mdl))
#     time_decod = SlidingEstimator(clf, n_jobs=n_jobs, scoring=scoring, verbose=True)
#     time_decod.fit(X, y)
#     coef = get_coef(time_decod, 'patterns_', inverse_transform=True)
#     evoked_time_gen = mne.EvokedArray(coef, info, tmin=times[0])
#     joint_kwargs = dict(ts_args=dict(time_unit='ms'), topomap_args=dict(time_unit='ms'))
#     evoked_time_gen.plot_joint(times=times, title='patterns', **joint_kwargs)




def cross_time_cond_gen(X_train, X_test, y_train, y_test, mdl, scoring, n_jobs=1):
    '''Train on X_train, test on X_test with generalization accross all time points'''
    # create model
    clf = make_pipeline(StandardScaler(),  mdl)
    time_gen = GeneralizingEstimator(clf, scoring=scoring, n_jobs=n_jobs, verbose=True)
    # Fit
    time_gen.fit(X=X_train, y=y_train)
    # Score on other condition
    scores = time_gen.score(X=X_test, y=y_test)

    return scores



def temporal_gen(X, y, mdl, scoring, cv=5, n_jobs=1):
    '''
    Temporal decoding with generalization accross time.
    Scores calculated using cross_val_multiscore: use simple models (e.g LogisticRegression
    as opposed to LogisticRegressionCV).


    X : np.array wit shape (n_trials, n_sensors, n_times)
    y : np.array
    mdl : sklearn model
    scoring: str, scoring method from sklearn

    returns
    scores : np.array
             Score at each time point, shape (n_times x n_times).
             The diagonal is the time by time decoding: np.diag(scores)


    '''

    clf = make_pipeline(StandardScaler(),  mdl) # create the model and pipeline
    time_gen = GeneralizingEstimator(clf, n_jobs=n_jobs, scoring=scoring, verbose=True)
    scores = cross_val_multiscore(time_gen, X, y, cv=cv, n_jobs=n_jobs)
    scores = np.mean(scores, axis=0) # Mean scores across cross-validation splits

    return scores



def plot_generalization_scores(scores, fname, times, vmin=0, vmax=1, annot=True, title='', num_ticks=20):
    '''
    annot : write values in matrix cells.
    '''
    fig, ax = plt.subplots(1)
    sns.heatmap(scores, annot=annot, cmap='YlOrRd', vmin=vmin, vmax=vmax,
                xticklabels=times, yticklabels=times,
                ax=ax)

    plt.locator_params(nbins=num_ticks)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor") # to rotate ticks
    plt.setp(ax.get_yticklabels(), rotation=0, ha="right", rotation_mode="anchor") # to rotate ticks
    # ---- add titles
    ax.set_xlabel('Testing Time (s)')
    ax.set_ylabel('Training Time (s)')
    ax.set_title(title)
    if fname:
        fig.savefig(fname)
        plt.close()


# ================== for IDEOGRAM study =================== #
def temporal_decoding_ideogram(X_words, X_images, y_words, y_images, mdl, scoring, cv=5, n_jobs=1):
    ''' Seperately decode y for words and images'''
    # create the model and pipeline
    clf = make_pipeline(StandardScaler(),  mdl)
    time_decod = SlidingEstimator(clf, n_jobs=n_jobs, scoring=scoring, verbose=True)
    scores_words = cross_val_multiscore(time_decod, X_words, y_words, cv=cv, n_jobs=n_jobs)
    scores_images = cross_val_multiscore(time_decod, X_images, y_images, cv=cv, n_jobs=n_jobs)

    # Mean scores across cross-validation splits
    scores_images = np.mean(scores_images, axis=0)
    scores_words = np.mean(scores_words, axis=0)

    return {'words':scores_words, 'images':scores_images}



def plot_temporal_scores_ideogram(scores_words, scores_images, times, fname, chance_level=None, figsize=[12,4]):
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(times, scores_words, label='words')
    ax.plot(times, scores_images, label='images')
    if chance_level:
        ax.axhline(chance_level, color='k', linestyle='--', label='chance')

    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Score')
    ax.legend()
    ax.set_title('Sensor space decoding')
    fig.savefig(fname)
    plt.close()
# =========================================================  #


# =========================== searchlight ==================================== #
def make_spatial_patches(epochs, patch_size):
    '''
    Make spatial patches using searchlight in sensor space.

    Parameters
    ------------
    epochs : mne.Epochs()
             The epochs object.
    patch_size : int
                 The size of the searchlight, unit is number of sensors.
                 The closest patch_size number of sensors will be included in each
                 patch.
                 Using n_sensors as a unit instead of a distance (e.g. 5mm) allows
                 to keep equal # of sensors in each patch in order to fit the same
                 classifier to all patches.

    Returns
    ------------
    patches_chNames : np.array (n_channels x patch_size)
                      Names of the channels in each patch.
                      Patch seed is the first channel in each patch
    patches_chLocs : np.array
                     Coordinates of the channels in each patch
    patches_dist : np.array
                   Distances of channels from the seed channel in each patch


    Channel order is the same as in epochs.info['chs'], usually MEG 001, MEG 002, ..., MEG 157
    To get all patch seeds : patches_chNames[:,0]
    '''

    # output lists
    patches_chNames = []
    patches_chLocs = []
    patches_dist = []

    # Compute the distances between the sensors
    locs = np.vstack([ch['loc'][:3] for ch in epochs.info['chs']]) # sensor coordinates
    dist = distance.squareform(distance.pdist(locs)) # matrix of distances (n_sensor x n_sensors)
    ch_names = np.array([ch['ch_name'] for ch in epochs.info['chs']]) # channel names

    for idx, chName in enumerate(ch_names):
        sub_dist = dist[idx, :] # distances from all sensors, idx=0 is self
        idx_neighbors = sub_dist.argsort()[:patch_size] # get idx of closest sensors, including self

        # get coordinates and sensor names of current patch and append to output lists
        patches_chLocs.append(locs[idx_neighbors])
        patches_chNames.append(ch_names[idx_neighbors])
        patches_dist.append(sub_dist[idx_neighbors])

    return np.array(patches_chNames), np.array(patches_chLocs), np.array(patches_dist)



def make_temporal_patches(epochs, patch_size):
    '''
    Generate temporal patches_times using searchlight.

    Parameters
    ------------
    epochs : mne.Epochs()
             The epochs object.
    patch_size : int
                Temporal radius, in samples.
                This defines how many neighboring time points to include in the
                time window.
                e.g. if patch_size = 10 : the patches_times will consist of the 10
                samples before and after each seed time-point
                Note, patch size will always end up being an odd number.

    Returns
    ------------
    patches_times : list of tuples
                    Each tuple is (tmin, tmax) of each time patch.
                    Epochs can then be cropped according to each patch later on.
    patches_idx : list of tuples
                  Each tuple is the index of (tmin, tmax) in epochs.times
                  of each time patch.
                  This is useful if need to select time slices from data that is
                  alraedy converted to an np.array() in previous steps of the
                  preprocessing.
    time_seeds : np.array
                Time points around which each temporal patch was created.
                Useful if need to append results of a tempral search back to
                the seed time points in an epochs object for e.g.


    '''

    n_times = len(epochs.times) # total number of time samples
    time_seeds = epochs.times[patch_size : n_times-patch_size] # Get valid seed time-points given the size of the sliding windows

    patches_times = [] # output list containg (tmnin, tmax) of each time patch
    patches_idx = [] # output list containg the index of (tmin, tmax) of each time patch

    for time in time_seeds:
        idx_time = np.where(epochs.times == time)[0][0] # get idx of seed point from the original time series epochs.times

        idx_tmin = idx_time - patch_size # idx of tmin of the slice : 10 samples before the seed
        idx_tmax = idx_time + patch_size # idx of tmin of the slice : 10 samples before the seed

        timeSlice = (epochs.times[idx_tmin], epochs.times[idx_tmax]) # time slice in milliseconds
        timeSlice_idx = (idx_tmin, idx_tmax) # indices of timepoints in epochs.times

        patches_times.append(timeSlice) # append to output list
        patches_idx.append(timeSlice_idx)

    return patches_times, patches_idx, time_seeds


def spatiotemporal_searchlight_decoding(epochs, y, tmin, tmax, spatial_radius, temporal_radius, mdl, scoring, cv, n_jobs, decim=1):
    '''
    Decoding over spatial and temporal patches.
    Unlike searchlight_decoding_slidingAverage(), no averaging is done over time
    windows, time dimension kept. Meaning a sliding time window of size temporal_radius
    is used at each time point. The time dimension of the time window is preserved.

    Parameters
    ----------

    epochs : mne.Epochs

    y : np.array

    tmin : float
           Min bound of temporal search

    tmax : float
            Max bound of temporal search

    spatial_radius : int
                      The size of the searchlight, unit is number of sensors.
                      See make_spatial_patches()

    temporal_radius: int
                     Temporal radius, in samples.
                     See make_temporal_patches()

    mdl : sklearn model, can be clf.

    scoring : str
              Sklearn scoring param.

    cv : int
         Number of cross validation

    n_jobs : int

    decim : int
            Decim value for resampling using epochs.decimate()


    Returns
    ----------
    all_scores_evoked : mne.Evoked
                        Scores at each sensor, at each time stored in an evoked
                        object

    '''
    # ---------------------- Subset epochs by selcted times ---------------------- #
    epochs.crop(tmin, tmax);

    if decim:
        print('Decimating using resampling (no sliding average)')
        epochs.decimate(decim)

    # ------------------ Create spatial and temoporal patches  ------------------- #
    spatial_patches, _ , _ = make_spatial_patches(epochs, spatial_radius)
    temporal_patches, _ , time_seeds = make_temporal_patches(epochs, temporal_radius)

    # ------------------------- Create output object  ---------------------------- #
    out_shape = (epochs.get_data().shape[1], epochs.get_data().shape[2]) # sensor x times
    all_scores = np.zeros(out_shape)

    # Loop over spatial and temoporal searchlight, tqdm adds progress bar
    for idx_ch, patch_ch in tqdm(enumerate(spatial_patches), total=spatial_patches.shape[0], desc='Spatial Searchlight'): # patch seeds are ordered MEG 001, MEG 002, ...

        for timeWindow, timeSeed in tqdm(zip(temporal_patches, time_seeds), total=len(temporal_patches), desc='Temporal Searchlight'):
            X = epochs.copy().pick_channels(patch_ch) # Subset to spatial patch

            clf = make_pipeline(Scaler(X.info), # needs to refit this for each patch if using mne.decoding.Scaler()
                                Vectorizer(), # Transform n-dimensional array into 2D array of n_samples by n_features.
                                mdl)

            patch_tmin = timeWindow[0] # get tmin and tmax of current time patch
            patch_tmax = timeWindow[1]

            X.crop(patch_tmin, patch_tmax) # crop to time patch
            X = X.get_data() # shape is (ntrial x nsensor x ntimes)

            # get scores with cross validation
            scores = cross_val_score(estimator=clf,
                                     X=X, y=y,
                                     scoring=scoring, cv=cv,
                                     n_jobs=n_jobs)

            # Get the idx in the original epochs.times which corresponds to current time window seed.
            # Necessary because the time patches indices dont align with those of epochs.times, since they have differnt dimensions
            # (some times are dropped because of edge problem)
            out_idx_time = np.where(epochs.times == timeSeed)[0][0]
            all_scores[idx_ch, out_idx_time] = scores.mean() # append to seed channel, seed time


    all_scores_evoked = mne.EvokedArray(all_scores, epochs.info) # convert to evoked object

    return all_scores_evoked



def searchlight_decoding_slidingAverage(epochs, y, mdl, scoring, cv, patches_chNames, times, decim, n_jobs):
    '''
    Special case implementaion of spatiotemporal_searchlight_decoding().
    Fit model at each of the searchlight patches. Stores result in evoked object.

    (1) Apply sliding averaging, if decim not None (see sliding_average())
    (2) For each spatial patch : apply temporal_decoding() i.e. decode at each time point


    Unlike spatiotemporal_searchlight_decoding(), here the mdl is trained/tested at
    each individual time window (time dimension = 1) as opposed to using a sliding
    time window of size>=1 where the time dimension is preserved.

    This function is a wrapper on temporal_decoding().

    Parameters
    ----------

    epochs : mne.Epochs
    y : np.array
    scoring : str
              sklearn decoding parameter e.g. 'accuracy'
    cv : int
         cross validation
    patches_chNames : np.array
                      List of channel names in each patch, generated using make_spatial_patches()
    decim : int
            If not none, apply sliding average
    n_jobs : int


    Returns
    ----------
    evoked_scores : mne.Evoked
                    Scores at each sensor, for each time point.

    '''


    scores = [] # final shape should be n_sensors x n_times
    for idx, patch_ch in enumerate(patches_chNames):
        print(idx+1, '/', len(patches_chNames))

        # patch_seed = patch_ch[0] # get patch seed
        X = epochs.copy().pick_channels(patch_ch) # Subset to patch

        if decim:
            print('Applying sliding average')
            X = sliding_average(X, decim)

        # Temporal decoding at patch
        scores_patch = temporal_decoding(X, y, mdl, scoring=scoring, cv=cv, n_jobs=n_jobs)
        scores.append(scores_patch) # append to output list

    evoked_scores = mne.EvokedArray(np.array(scores), epochs.info) # store results in epochs object
    evoked_scores.times = times

    return evoked_scores


# =================================== RSA ==================================== #
def make_rdm_from_1darray(arr, metric='correlation'):
    '''1d array to rdm'''
    flat_distances = scipy.spatial.distance.pdist(np.atleast_2d(arr).T, metric=metric)
    rdm = scipy.spatial.distance.squareform(flat_distances)
    return rdm

def make_rdm_from_2darray(arr, metric='correlation'):
    '''
    2d array to rdm.

    arr : 2D array (e.g. trials x sensors; trials x word2vec_features). RDMs over first dimension (trials)
    (Currently using this by subsetting at a specific time point and getting rdm accross trials and sensors)
    '''
    flat_distances = scipy.spatial.distance.pdist(arr, metric=metric)
    rdm = scipy.spatial.distance.squareform(flat_distances)
    return rdm

def correlate_model_and_rdm_timecourses(rdm_tc, model_rdm):
    '''
    Correlate model RDM with timecourse of RDMs, at each time-point

    rdm_tc : np.array
             one rdm per time point, shape is (ntrials, ntrials, ntimes)

    model_rdm : np.array (ntrials x ntrials)


    returns
    ----------
    corrs : Shape ntimes for orrelation coefficients of the rdms
    pvals : Shape ntimes for pvals of the correlation coefficients of the rdms

    '''
    # calculate correlation accross words and images, at each time point
    corrs = np.zeros(rdm_tc.shape[2])
    pvals = np.zeros(rdm_tc.shape[2])

    ntimes = rdm_tc.shape[2]
    rdm_tc_flat = rdm_tc.reshape(-1,ntimes)
    model_rdm_flat = model_rdm.reshape(-1)

    for i in range(ntimes):
        sub_rdm = rdm_tc_flat[:,i]
        corrs[i], pvals[i] = pearsonr(sub_rdm, model_rdm_flat)

    return corrs, pvals


def partial_correlation_model_and_rdm_timecourses(rdm_tc, model1_rdm, model2_rdm, method='pearson'):
    '''
    Partial correlation of model1 RDM with timecourse of RDMs, at each time-point
    after partialling out model2 RDM.

    rdm_tc : np.array
             one rdm per time point, shape is (ntrials, ntrials, ntimes)

    model1_rdm : np.array (ntrials x ntrials)
    model2_rdm : np.array (ntrials x ntrials)

    method: str
            Correlation type (see pingouin.partial_corr)

    returns
    ----------
    corrs : Shape ntimes for orrelation coefficients of the rdms
    pvals : Shape ntimes for pvals of the correlation coefficients of the rdms

    '''
    # calculate correlation accross words and images, at each time point
    corrs = np.zeros(rdm_tc.shape[2])
    pvals = np.zeros(rdm_tc.shape[2])

    ntimes = rdm_tc.shape[2]
    rdm_tc_flat = rdm_tc.reshape(-1,ntimes)
    model1_rdm_flat = model1_rdm.reshape(-1)
    model2_rdm_flat = model2_rdm.reshape(-1)

    for i in range(ntimes):
        sub_rdm = rdm_tc_flat[:,i]
        df = pd.DataFrame({'sub_rdm':sub_rdm,
                           'model1_rdm_flat':model1_rdm_flat,
                           'model2_rdm_flat':model2_rdm_flat})

        stats = partial_corr(df, x='sub_rdm', y='model1_rdm_flat', y_covar='model2_rdm_flat', method=method)
        corrs[i] = stats['r']
        pvals[i] = stats['p-val']

    return corrs, pvals


def correlate_rdm_timecourses(rdm_tc1, rdm_tc2):
    '''
    Correlate 2 timecourses of RDMs, usually empirical RDMs from epochs/evokeds

    rdm_tc1, rdm_tc2 : np.array
                      one rdm per time point, shape is (ntrials, ntrials, ntimes)

    returns
    ----------
    corrs : ntimes x ntimes correlation coefficients of the rdms
    pvals : ntimes x ntimes pvals of the correlation coefficients of the rdms

    '''
    # calculate correlation accross words and images, at each time point
    corrs = np.zeros([rdm_tc1.shape[2], rdm_tc2.shape[2]])
    pvals = np.zeros([rdm_tc1.shape[2], rdm_tc2.shape[2]])

    ntimes = rdm_tc1.shape[2]
    rdm1_flat = rdm_tc1.reshape(-1,ntimes)
    rdm2_flat = rdm_tc2.reshape(-1,ntimes)

    for i in range(ntimes):
        sub_rdm1 = rdm1_flat[:,i]
        for j in range(ntimes):
            sub_rdm2 = rdm2_flat[:,j]

            corrs[i,j], pvals[i,j] = pearsonr(sub_rdm1, sub_rdm2)

    return corrs, pvals



def rsa_anderson_etal_2016(data_rdm, model_rdm, verbose=True):
    '''
    Decoding using the RSA methods in:
    Anderson, A. J., Zinszer, B. D., & Raizada, R. D. (2016). Representational similarity encoding for fMRI: Pattern-based synthesis to predict brain activity using stimulus-model-similarities. NeuroImage, 128, 44-53.

    data_rdm : np.array of size (n_exemplars x n_exemplars)
    model_rdm : np.array of size (n_exemplars x n_exemplars)

    returns
    --------
    accuracy : float
               Percentage accuracy over all possible pairs


    '''
    assert data_rdm.shape == model_rdm.shape, 'Model and data rdm do not have the same shape'

    # Get all possible pairs of concepts
    conc_pairs_idx = list(combinations(range(data_rdm.shape[0]), 2 ))
    hits = [] # will count correctly decoded pairs

    # Counter for printing progress bar
    n_comparisons = len(conc_pairs_idx)
    counter = 0


    for idx, pair in enumerate(conc_pairs_idx):
        counter += 1
        if verbose: print(counter, '/', n_comparisons, end='\r')

        # (1) ----- Extract brain and model vector from dsms
        model_1 = model_rdm[:, pair[0]]
        model_2 = model_rdm[:, pair[1]]

        brain_1 = data_rdm[:, pair[0]]
        brain_2 = data_rdm[:, pair[1]]

        # extract labels, in case needed
        #label_1 = exemplar_labels[pair[0]] # can also use lexical_stats.concept. Same order
        #label_2 = exemplar_labels[pair[1]]

        # (2) ----- Get rid of the element that represents the test items
        model_1 = np.delete(model_1, pair)
        model_2 = np.delete(model_2, pair)

        brain_1 = np.delete(brain_1, pair)
        brain_2 = np.delete(brain_2, pair)

        # (3) Correlate the model and brain vectors.
        # here a hit is when corr(brain1, model1) + corr(brain2, model2) > corr(brain1, model2) + corr(brain2, model1)
        # Note the correlations are first z transformed using np.arctanh

        true_pairing_corr = arctanh(pearsonr(brain_1, model_1))[0] + arctanh(pearsonr(brain_2, model_2))[0]
        shuffled_pairing_corr = arctanh(pearsonr(brain_1, model_2))[0] + arctanh(pearsonr(brain_2, model_1))[0]

        if true_pairing_corr > shuffled_pairing_corr:
            hits.append(1)

    # compute accuracy over all possible pairs
    accuracy = sum(hits) / len(conc_pairs_idx)

    return accuracy


# ===================== Regression ============================= #
def fit_regression_and_plot(epochs_object, design, target_type, plotting=False, fig_dir=None):

    '''
    Quick regression with no correction accross subjects. Use this for sanity checks with single subjects

    epochs_object : mne.Epochs
    design : pd.DataFrame with columns as the factors
    target_type : str, used for plot title and fname only
    plotting : bool
    fig_dir : str, directory to save figures

    '''
    var_names = design.columns
    res = linear_regression(epochs_object, design_matrix=design, names=var_names)
    if plotting:
        for cond in var_names:
            print(cond)
            # plot the resulting coefficients (Beta Values
            fig1 = res[cond].beta.plot_joint(title='%s %s'%(cond, target_type), ts_args=dict(time_unit='ms'), topomap_args=dict(time_unit='ms'), show=False)
            fig1.savefig(fig_dir + 'coeffs_%s_%s.png'%(cond, target_type))
            plt.close()
            # check for significance

            reject_H0, fdr_pvals = fdr_correction(res[cond].p_val.data)
            evoked = res[cond].beta
            fig2 = evoked.plot_image(mask=reject_H0, time_unit='ms', show=False)
            fig2.savefig(fig_dir + 'pvals_%s_%s.png'%(cond, target_type))
            plt.close()
    return res
