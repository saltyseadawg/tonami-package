# Modified cross_validate code from skelarn.model_selection.cross_validate
# Needed to modify to get 
# - average explained variances from preprocessing
# - train and test distributions of best estimator

import time
from collections import Counter

import numpy as np
from joblib import Parallel

from sklearn.base import is_classifier, clone
from sklearn.utils.fixes import delayed
from sklearn.utils.metaestimators import _safe_split
from sklearn.metrics import check_scoring
from sklearn.model_selection import check_cv
from sklearn.model_selection._validation import _aggregate_score_dicts

SCORING='balanced_accuracy'

def _get_data_set_stats(y):
    '''
    Returns y's distribution percentage with labels and length of y
    '''
    hist = Counter(y)
    num = len(y)
    dist = [(i, hist[i] / num * 100.0) for i in hist]
    dist = sorted(dist, key=lambda tup: tup[0])
    dist = [dist[0][1], dist[1][1], dist[2][1], dist[3][1]]
    dist = np.around(dist, 2)
    return dist, num

def cross_validate_tonami(estimator, X, y, cv=None, n_jobs=None):
    '''
    Evaluate model by cross-validation and also record fit/score times. Customized to only do balanced accuracy.
    '''
    cv = check_cv(cv, y, classifier=is_classifier(estimator))
    scorer = check_scoring(estimator, SCORING)

    parallel = Parallel(n_jobs=n_jobs, verbose=0, pre_dispatch="2*n_jobs")
    results_indiv = parallel(
        delayed(_fit_and_score)(clone(estimator), X, y, scorer, train, test)
        for train, test in cv.split(X, y)
    )

    results = _aggregate_score_dicts(results_indiv)

    ret = {
        "test_score_stats": {
            "mean": np.mean(results["test_score"]),
            "std":  np.std( results["test_score"]),
            "min":  np.amin(results["test_score"]),
            "max":  np.amax(results["test_score"]),
        },
        "train_score": np.mean(results["train_score"]),
        "fit_time": {
            "mean_total":       np.mean(results["fit_time"]),
            "mean_per_sample":  np.mean(np.divide(results["fit_time"], results["n_train_samples"])),
        },
        "score_time": {
            "mean_total":       np.mean(results["score_time"]),
            "mean_per_sample":  np.mean(np.divide(results["score_time"], results["n_test_samples"])),
        },
        "explained_variance":   np.mean(results["explained_variance"]),
        "best_estimator_dict":  results_indiv[np.argmax(results["test_score"])],
    }

    return ret

def _fit_and_score(estimator, X, y, scorer, train, test):
    '''
    Fit estimator and compute scores for a given dataset split.
    Times are in seconds.
    '''
    start_time = time.time()

    X_train, y_train = _safe_split(estimator, X, y, train)
    X_test, y_test = _safe_split(estimator, X, y, test, train)

    estimator.fit(X_train, y_train)

    fit_time = time.time() - start_time
    test_scores = scorer(estimator, X_test, y_test)
    score_time = time.time() - start_time - fit_time
    train_scores = scorer(estimator, X_train, y_train)
    
    n_model_features = estimator.named_steps['estimator'].n_features_in_
    preprocessing = estimator.named_steps["preprocessing"]

    result = {
        "test_score": test_scores,
        "train_score": train_scores,
        "fit_time": fit_time,
        "score_time": score_time,
        "estimator": estimator,
        "y_test": y_test,
        "y_pred": estimator.predict(X_test),
        "n_model_features": n_model_features,
        "n_segment_features": n_model_features  if preprocessing is None else preprocessing.n_features_in_,
        "explained_variance": 1.0               if preprocessing is None else np.sum(preprocessing.explained_variance_ratio_),
    }
    result["train_dist"],   result["n_train_samples"] = _get_data_set_stats(y_train)
    result["test_dist"],    result["n_test_samples"] =  _get_data_set_stats(y_test)

    return result
