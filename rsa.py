import scipy
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from scipy.spatial import distance
from itertools import combinations
from pingouin import partial_corr
from numpy import arctanh


def make_rdm_from_1darray(arr, metric='correlation'):
    '''1d array to rdm'''
    flat_distances = distance.pdist(np.atleast_2d(arr).T, metric=metric)
    rdm = distance.squareform(flat_distances)
    return rdm

def make_rdm_from_2darray(arr, metric='correlation'):
    '''
    2d array to rdm.

    arr : 2D array (e.g. trials x sensors; trials x word2vec_features). RDMs over first dimension (trials)
    (Currently using this by subsetting at a specific time point and getting rdm accross trials and sensors)
    '''
    flat_distances = distance.pdist(arr, metric=metric)
    rdm = distance.squareform(flat_distances)
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
