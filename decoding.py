import mne
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from mne.decoding import GeneralizingEstimator, SlidingEstimator, cross_val_multiscore, Vectorizer, Scaler
from mne.stats import linear_regression, fdr_correction
from scipy.spatial import distance
from tqdm import tqdm

from .preproc import sliding_average


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



def cross_time_cond_genCV(X_train, X_test, y_train, y_test, mdl, scoring, cv=5, n_jobs=1):
    '''
    Train on X_train, test on X_test with generalization accross all time points
    Cross validation done by training on a subset of X_train  and testing
    on a subset of X_test, with no repeats of trials accross the train-test split.
    Order of trials/exemplars in X_train, X_test, y_train, y_test must be the same.
    '''
    # create model
    clf = make_pipeline(StandardScaler(),  mdl)
    time_gen = GeneralizingEstimator(clf, scoring=scoring, n_jobs=n_jobs, verbose=True)
    kf = KFold(n_splits=cv)

    scores = []
    for train_index, test_index in kf.split(X_train):
        fold_X_train = X_train[train_index]
        fold_X_test = X_test[test_index]

        fold_y_train = y_train[train_index]
        fold_y_test = y_test[test_index]

        # Fit
        time_gen.fit(X=fold_X_train, y=fold_y_train)
        # Score on other condition
        fold_scores = time_gen.score(X=fold_X_test, y=fold_y_test)


        scores.append(fold_scores)

    scores = np.array(scores)

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


# ==================================== regression ============================ #
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
