'''
See:
Hultén, A., van Vliet, M., Kivisaari, S., Lammi, L., Lindh‐Knuutila, T., Faisal, A., & Salmelin, R. (2021). The neural representation of abstract words may arise through grounding word meaning in language itself. Human brain mapping, 42(15), 4973-4984.
Palatucci, M., Pomerleau, D., Hinton, G. E., & Mitchell, T. M. (2009). Zero- shot learning with semantic output codes (pp. 1410–1418). Pittsburgh, PA: Carnegie Mellon University.
Sudre, G., Pomerleau, D., Palatucci, M., Wehbe, L., Fyshe, A., Salmelin, R., & Mitchell, T. (2012). Tracking neural coding of perceptual and semantic features of concrete nouns. NeuroImage, 62(1), 451-463.

'''

from itertools import combinations
from joblib import Parallel, delayed
from scipy.spatial.distance import cosine
from tqdm import tqdm
import numpy as np
import os


def zeroshot_classification(X, y, clf):
    ''''
    Zero-shot classification.
    Looping over time dimension not implemented within function.

    PIPELINE:
    Leave-2-out split:
        Train classifier
        Predict 2 vectors of leave2out
        Compare predicted vectors (p1, p2) to model vectors (s1,s2) using cosine distance:
            IF  distance(p1,s1) + distance(p2,s2) < distance(p1,s2) + distance(p2,s1)
                    HIT
            ELSE
                    MISS

    Repeat for all possible pairs
    '''


    n_trials = X.shape[0]
    indices_pairs = list(combinations( range(n_trials), 2))

    hits = []
    for idx_leave2out in tqdm(indices_pairs): # leave 2 out loop, all possible pairs

        # Train-test-split using leave2out
        X_train = np.delete(X, idx_leave2out, axis=0)
        X_test = X[idx_leave2out,:]

        y_train = np.delete(y, idx_leave2out, axis=0)
        y_test = y[idx_leave2out,:]

        # fit classifier on training data
        clf.fit(X_train, y_train)

        # predicter 2 left out
        preds = clf.predict(X_test)

        hit = cosine(preds[0],y_test[0]) + cosine(preds[1],y_test[1]) < cosine(preds[0],y_test[1]) + cosine(preds[1],y_test[0])

        if hit:
            hits.append(1)
        else:
            hits.append(0)

    accuracy = sum(hits) / len(hits)

    return accuracy



def _leave2out(idx_leave2out, X, y):
    '''
    Called by zeroshot_classification_parallelJobs.

    '''

    # Train-test-split using leave2out
    X_train = np.delete(X, idx_leave2out, axis=0) # select the training data
    X_test = X[idx_leave2out,:]

    y_train = np.delete(y, idx_leave2out, axis=0) # select testing data
    y_test = y[idx_leave2out,:]

    # fit classifier on training data
    clf.fit(X_train, y_train)

    # predicter 2 left out
    preds = clf.predict(X_test)

    hit = cosine(preds[0],y_test[0]) + cosine(preds[1],y_test[1]) < cosine(preds[0],y_test[1]) + cosine(preds[1],y_test[0])

    if hit:
        return 1
    else:
        return 0


def zeroshot_classification_parallelJobs(X, y, clf, n_jobs):
    ''''
    Same as zeroshot_classification but 5-6x faster.
    '''


    n_trials = X.shape[0]
    indices_pairs = list(combinations( range(n_trials), 2))

    hits =  Parallel(n_jobs=n_jobs)(delayed(_leave2out)(i, X, y)  for i in indices_pairs)
    accuracy = sum(hits) / len(hits)

    return accuracy



def zeroshot_classification_condGen(X_train, X_test, y_train, y_test):

    ''''
    Zero-shot classification with condition generalization.
    Looping over time dimension not implemented within function.

    PIPELINE:

    Train classifier on training condition
    For all possible pairs of testing condition
        Predict vectors
        Compare predicted vectors (p1, p2) to model vectors (s1,s2) using cosine distance:
            IF  distance(p1,s1) + distance(p2,s2) < distance(p1,s2) + distance(p2,s1)
                    HIT
            ELSE
                    MISS

    Repeat for all possible pairs
    Get accuracy accross all pairs
    '''
    # fit classifier on training data
    clf.fit(X_train, y_train)

    # create list of indices of all possible pairs
    n_trials = X_train.shape[0]
    indices_pairs = list(combinations( range(n_trials), 2))

    hits = []
    for idx_pairs in tqdm(indices_pairs): # loop through all possible pairs

        X_tested_pairs = X[idx_pairs,:]
        y_tested_pairs = y_test[idx_pairs,:]

        # predicter 2 left out
        preds = clf.predict(X_tested_pairs)

        hit = cosine(preds[0],y_tested_pairs[0]) + cosine(preds[1],y_tested_pairs[1]) < cosine(preds[0],y_tested_pairs[1]) + cosine(preds[1],y_tested_pairs[0])

        if hit:
            hits.append(1)
        else:
            hits.append(0)

    accuracy = sum(hits) / len(hits)

    return accuracy
