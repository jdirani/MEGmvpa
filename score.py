def score_correlation(y_test, y_preds):
    '''
    y_test :  shape = (n_exemplars, n_features)
    y_preds: shape = (n_exemplars, n_times, n_features)
    
    For each timepoint,
        For each prediciton,
            Correlate with true vector.
        Average over all predictions

    Returns timecourse of mean correlaiton over all predicted exemplars

    '''
    n_times = y_preds.shape[1]
    n_preds = y_preds.shape[0]

    corr_timecourse = [] # timecourse of mean correlations over all corr(y_pred, y_true). shape = n_time
    for tt in range(n_times):
        tt_corrs = [] # correlation at each time for all y_test,y_true pairs. Shape will be len(y_test)
        for pp in range(n_preds): # for each prediciton, correlate with true vector
            pred = y_preds[pp,tt,:]  # predicted vector at current time, for current exemplar
            true = y_test[pp,:] # y_true vector for current exemplar, same for all timepoints.
            r,p = pearsonr(pred, true) # correlate for current exemplar, current timepoint
            tt_corrs.append(r) # append to list containing correlation for all exemplars.

        tt_mean_corr = np.mean(tt_corrs) # calculate average correlation over exemplars, for current time tt
        corr_timecourse.append(tt_mean_corr) # append current time mean correlaiton to timecourse

    return corr_timecourse
