# Modified cross_validate code from skelarn.model_selection.cross_validate
# Needed to modify to get average explained variances from preprcossors


import warnings
import numbers
import time
from traceback import format_exc
from contextlib import suppress
from collections import Counter

import numpy as np
from joblib import Parallel, logger

from sklearn.base import is_classifier, clone
from sklearn.utils import indexable
from sklearn.utils.validation import _check_fit_params
from sklearn.utils.fixes import delayed
from sklearn.utils.metaestimators import _safe_split
from sklearn.metrics import check_scoring
from sklearn.metrics._scorer import _MultimetricScorer
from sklearn.exceptions import FitFailedWarning, NotFittedError
from sklearn.model_selection import check_cv


__all__ = [
    "cross_validate_tonami",
    "cross_validate",
]

def get_data_set_stats(y):
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


def cross_validate_tonami(
    estimator,
    X,
    y,
    cv=None,
    n_jobs=None,
):
    """Evaluate metric(s) by cross-validation and also record fit/score times.

    Read more in the :ref:`User Guide <multimetric_cross_validation>`.

    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.

    X : array-like of shape (n_samples, n_features)
        The data to fit. Can be for example a list, or an array.

    y : array-like of shape (n_samples,) or (n_samples, n_outputs), \
            default=None
        The target variable to try to predict in the case of
        supervised learning.

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 5-fold cross validation,
        - int, to specify the number of folds in a `(Stratified)KFold`,
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For int/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`.Fold` is used. These splitters are instantiated
        with `shuffle=False` so the splits will be the same across calls.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

        .. versionchanged:: 0.22
            ``cv`` default value if None changed from 3-fold to 5-fold.

    n_jobs : int, default=None
        Number of jobs to run in parallel. Training the estimator and computing
        the score are parallelized over the cross-validation splits.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    Returns
    -------
    scores : dict
        test_scores is a dict containing mean, std, min and max
        train_score, explained_variance are means across estimators
        fit_time, score_time are dict containing mean_total and mean_per_sample
        best_estimator is the estimator with the highest test_score

        A dict of arrays containing the score/time arrays for each scorer is
        returned. The possible keys for this ``dict`` are:

            ``test_score_stats``
                The dictionary containing ``mean``, ``std``, ``min``, and 
                ``max`` of all the test score on each cv split.
            ``train_score``
                The score array for train scores on each cv split.
                Suffix ``_score`` in ``train_score`` changes to a specific
                metric like ``train_r2`` or ``train_auc`` if there are
                multiple scoring metrics in the scoring parameter.
                This is available only if ``return_train_score`` parameter
                is ``True``.
            ``fit_time``
                The dictionary of times for fitting the estimator on the train
                set for each cv split. Contains ``mean_total`` which is the 
                mean of the total time it takes to fit each set. Contains 
                ``mean_per_sample`` which is the mean of the total time to fit 
                each set averaged across the number of samples in the train set.
            ``score_time``
                The dictionary of times for scoring the estimator on the test 
                set for each cv split. Contains ``mean_total`` which is the mean 
                of the total time it takes to fit each set. Contains 
                ``mean_per_sample`` which is the mean of the total time to fit 
                each set averaged across the number of samples in the train set.
            ``best_estimator_dict``
                The dictionary containing ``test_score``, ``train_score``, 
                ``fit_time``, ``score_time``, ``n_train_samples``, 
                ``n_test_samples``, ``train_dist``, ``test_dist``, ``y_test``, ``y_pred`` and 
                ``explained_variance`` of the 
                returned best_estimator

    """
    # original cross_validation parameters from sklearn I made constant
    groups = None
    scoring = 'balanced_accuracy'
    verbose=0
    fit_params=None
    pre_dispatch="2*n_jobs"
    return_train_score=True
    return_estimator=True
    error_score=np.nan

    X, y, groups = indexable(X, y, groups)

    cv = check_cv(cv, y, classifier=is_classifier(estimator))

    scorers = check_scoring(estimator, scoring)

    # We clone the estimator to make sure that all the folds are
    # independent, and that it is pickle-able.
    parallel = Parallel(n_jobs=n_jobs, verbose=verbose, pre_dispatch=pre_dispatch)
    results_indiv = parallel(
        delayed(_fit_and_score)(
            clone(estimator),
            X,
            y,
            scorers,
            train,
            test,
            verbose,
            None,
            fit_params,
            return_train_score=return_train_score,
            return_n_test_samples=True,
            return_times=True,
            return_estimator=return_estimator,
            error_score=error_score,
        )
        for train, test in cv.split(X, y, groups)
    )

    _warn_about_fit_failures(results_indiv, error_score)

    # For callabe scoring, the return type is only know after calling. If the
    # return type is a dictionary, the error scores can now be inserted with
    # the correct key.
    if callable(scoring):
        _insert_error_scores(results_indiv, error_score)

    results = _aggregate_score_dicts(results_indiv)

    best_est_idx = np.argmax(results["test_score"])

    ret = {
        "test_score_stats": {
            "mean": np.mean(results["test_score"]),
            "std": np.std(results["test_score"]),
            "min": np.amin(results["test_score"]),
            "max": np.amax(results["test_score"]),
        },
        "train_score": np.mean(results["train_score"]),
        "fit_time": {
            "mean_total": np.mean(results["fit_time"]),
            "mean_per_sample": np.mean(np.divide(results["fit_time"], results["n_train_samples"])),
        },
        "score_time": {
            "mean_total": np.mean(results["score_time"]),
            "mean_per_sample": np.mean(np.divide(results["score_time"], results["n_test_samples"])),
        },
        "explained_variance": np.mean(results["explained_variance"]),
        "best_estimator_dict": results_indiv[best_est_idx],
    }

    return ret


def _insert_error_scores(results, error_score):
    """Insert error in `results` by replacing them inplace with `error_score`.

    This only applies to multimetric scores because `_fit_and_score` will
    handle the single metric case.
    """
    successful_score = None
    failed_indices = []
    for i, result in enumerate(results):
        if result["fit_error"] is not None:
            failed_indices.append(i)
        elif successful_score is None:
            successful_score = result["test_scores"]

    if successful_score is None:
        raise NotFittedError("All estimators failed to fit")

    if isinstance(successful_score, dict):
        formatted_error = {name: error_score for name in successful_score}
        for i in failed_indices:
            results[i]["test_scores"] = formatted_error.copy()
            if "train_scores" in results[i]:
                results[i]["train_scores"] = formatted_error.copy()


def _warn_about_fit_failures(results, error_score):
    fit_errors = [
        result["fit_error"] for result in results if result["fit_error"] is not None
    ]
    if fit_errors:
        num_failed_fits = len(fit_errors)
        num_fits = len(results)
        fit_errors_counter = Counter(fit_errors)
        delimiter = "-" * 80 + "\n"
        fit_errors_summary = "\n".join(
            f"{delimiter}{n} fits failed with the following error:\n{error}"
            for error, n in fit_errors_counter.items()
        )

        some_fits_failed_message = (
            f"\n{num_failed_fits} fits failed out of a total of {num_fits}.\n"
            "The score on these train-test partitions for these parameters"
            f" will be set to {error_score}.\n"
            "If these failures are not expected, you can try to debug them "
            "by setting error_score='raise'.\n\n"
            f"Below are more details about the failures:\n{fit_errors_summary}"
        )
        warnings.warn(some_fits_failed_message, FitFailedWarning)


def _fit_and_score(
    estimator,
    X,
    y,
    scorer,
    train,
    test,
    verbose,
    parameters,
    fit_params,
    return_train_score=False,
    return_parameters=False,
    return_n_test_samples=False,
    return_times=False,
    return_estimator=False,
    split_progress=None,
    candidate_progress=None,
    error_score=np.nan,
):

    """Fit estimator and compute scores for a given dataset split.

    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.

    X : array-like of shape (n_samples, n_features)
        The data to fit.

    y : array-like of shape (n_samples,) or (n_samples, n_outputs) or None
        The target variable to try to predict in the case of
        supervised learning.

    scorer : A single callable or dict mapping scorer name to the callable
        If it is a single callable, the return value for ``train_scores`` and
        ``test_scores`` is a single float.

        For a dict, it should be one mapping the scorer name to the scorer
        callable object / function.

        The callable object / fn should have signature
        ``scorer(estimator, X, y)``.

    train : array-like of shape (n_train_samples,)
        Indices of training samples.

    test : array-like of shape (n_test_samples,)
        Indices of test samples.

    verbose : int
        The verbosity level.

    error_score : 'raise' or numeric, default=np.nan
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised.
        If a numeric value is given, FitFailedWarning is raised.

    parameters : dict or None
        Parameters to be set on the estimator.

    fit_params : dict or None
        Parameters that will be passed to ``estimator.fit``.

    return_train_score : bool, default=False
        Compute and return score on training set.

    return_parameters : bool, default=False
        Return parameters that has been used for the estimator.

    split_progress : {list, tuple} of int, default=None
        A list or tuple of format (<current_split_id>, <total_num_of_splits>).

    candidate_progress : {list, tuple} of int, default=None
        A list or tuple of format
        (<current_candidate_id>, <total_number_of_candidates>).

    return_n_test_samples : bool, default=False
        Whether to return the ``n_test_samples``.

    return_times : bool, default=False
        Whether to return the fit/score times.

    return_estimator : bool, default=False
        Whether to return the fitted estimator.

    Returns
    -------
    result : dict with the following attributes
        train_scores : dict of scorer name -> float
            Score on training set (for all the scorers),
            returned only if `return_train_score` is `True`.
        test_scores : dict of scorer name -> float
            Score on testing set (for all the scorers).
        n_test_samples : int
            Number of test samples.
        fit_time : float
            Time spent for fitting in seconds.
        score_time : float
            Time spent for scoring in seconds.
        parameters : dict or None
            The parameters that have been evaluated.
        estimator : estimator object
            The fitted estimator.
        fit_error : str or None
            Traceback str if the fit failed, None if the fit succeeded.
    """
    if not isinstance(error_score, numbers.Number) and error_score != "raise":
        raise ValueError(
            "error_score must be the string 'raise' or a numeric value. "
            "(Hint: if using 'raise', please make sure that it has been "
            "spelled correctly.)"
        )

    progress_msg = ""
    if verbose > 2:
        if split_progress is not None:
            progress_msg = f" {split_progress[0]+1}/{split_progress[1]}"
        if candidate_progress and verbose > 9:
            progress_msg += f"; {candidate_progress[0]+1}/{candidate_progress[1]}"

    if verbose > 1:
        if parameters is None:
            params_msg = ""
        else:
            sorted_keys = sorted(parameters)  # Ensure deterministic o/p
            params_msg = ", ".join(f"{k}={parameters[k]}" for k in sorted_keys)
    if verbose > 9:
        start_msg = f"[CV{progress_msg}] START {params_msg}"
        print(f"{start_msg}{(80 - len(start_msg)) * '.'}")

    # Adjust length of sample weights
    fit_params = fit_params if fit_params is not None else {}
    fit_params = _check_fit_params(X, fit_params, train)

    if parameters is not None:
        # clone after setting parameters in case any parameters
        # are estimators (like pipeline steps)
        # because pipeline doesn't clone steps in fit
        cloned_parameters = {}
        for k, v in parameters.items():
            cloned_parameters[k] = clone(v, safe=False)

        estimator = estimator.set_params(**cloned_parameters)

    start_time = time.time()

    X_train, y_train = _safe_split(estimator, X, y, train)
    X_test, y_test = _safe_split(estimator, X, y, test, train)

    result = {}
    try:
        if y_train is None:
            estimator.fit(X_train, **fit_params)
        else:
            estimator.fit(X_train, y_train, **fit_params)

    except Exception:
        # Note fit time as time until error
        fit_time = time.time() - start_time
        score_time = 0.0
        if error_score == "raise":
            raise
        elif isinstance(error_score, numbers.Number):
            if isinstance(scorer, dict):
                test_scores = {name: error_score for name in scorer}
                if return_train_score:
                    train_scores = test_scores.copy()
            else:
                test_scores = error_score
                if return_train_score:
                    train_scores = error_score
        result["fit_error"] = format_exc()
    else:
        result["fit_error"] = None

        fit_time = time.time() - start_time
        test_scores = _score(estimator, X_test, y_test, scorer, error_score)
        score_time = time.time() - start_time - fit_time
        train_scores = _score(estimator, X_train, y_train, scorer, error_score)

    if verbose > 1:
        total_time = score_time + fit_time
        end_msg = f"[CV{progress_msg}] END "
        result_msg = params_msg + (";" if params_msg else "")
        if verbose > 2:
            if isinstance(test_scores, dict):
                for scorer_name in sorted(test_scores):
                    result_msg += f" {scorer_name}: ("
                    if return_train_score:
                        scorer_scores = train_scores[scorer_name]
                        result_msg += f"train={scorer_scores:.3f}, "
                    result_msg += f"test={test_scores[scorer_name]:.3f})"
            else:
                result_msg += ", score="
                if return_train_score:
                    result_msg += f"(train={train_scores:.3f}, test={test_scores:.3f})"
                else:
                    result_msg += f"{test_scores:.3f}"
        result_msg += f" total time={logger.short_format_time(total_time)}"

        # Right align the result_msg
        end_msg += "." * (80 - len(end_msg) - len(result_msg))
        end_msg += result_msg
        print(end_msg)

    n_model_features = estimator.named_steps['estimator'].n_features_in_
    preprocessing = estimator.named_steps["preprocessing"]

    result["test_score"] = test_scores   
    result["train_score"] = train_scores 
    result["n_model_features"] = n_model_features
    result["n_segment_features"] = n_model_features if preprocessing is None else preprocessing.n_features_in_
    result["train_dist"], result["n_train_samples"] = get_data_set_stats(y_train)
    result["test_dist"], result["n_test_samples"] = get_data_set_stats(y_test)
    result["fit_time"] = fit_time
    result["score_time"] = score_time
    result["estimator"] = estimator
    result["explained_variance"] = 1.0 if preprocessing is None else np.sum(preprocessing.explained_variance_ratio_)
    result["y_test"] = y_test
    result["y_pred"] = estimator.predict(X_test)
    return result


def _score(estimator, X_test, y_test, scorer, error_score="raise"):
    """Compute the score(s) of an estimator on a given test set.

    Will return a dict of floats if `scorer` is a dict, otherwise a single
    float is returned.
    """
    if isinstance(scorer, dict):
        # will cache method calls if needed. scorer() returns a dict
        scorer = _MultimetricScorer(**scorer)

    try:
        if y_test is None:
            scores = scorer(estimator, X_test)
        else:
            scores = scorer(estimator, X_test, y_test)
    except Exception:
        if error_score == "raise":
            raise
        else:
            if isinstance(scorer, _MultimetricScorer):
                scores = {name: error_score for name in scorer._scorers}
            else:
                scores = error_score
            warnings.warn(
                "Scoring failed. The score on this train-test partition for "
                f"these parameters will be set to {error_score}. Details: \n"
                f"{format_exc()}",
                UserWarning,
            )

    error_msg = "scoring must return a number, got %s (%s) instead. (scorer=%s)"
    if isinstance(scores, dict):
        for name, score in scores.items():
            if hasattr(score, "item"):
                with suppress(ValueError):
                    # e.g. unwrap memmapped scalars
                    score = score.item()
            if not isinstance(score, numbers.Number):
                raise ValueError(error_msg % (score, type(score), name))
            scores[name] = score
    else:  # scalar
        if hasattr(scores, "item"):
            with suppress(ValueError):
                # e.g. unwrap memmapped scalars
                scores = scores.item()
        if not isinstance(scores, numbers.Number):
            raise ValueError(error_msg % (scores, type(scores), scorer))
    return scores


def _aggregate_score_dicts(scores):
    """Aggregate the list of dict to dict of np ndarray

    The aggregated output of _aggregate_score_dicts will be a list of dict
    of form [{'prec': 0.1, 'acc':1.0}, {'prec': 0.1, 'acc':1.0}, ...]
    Convert it to a dict of array {'prec': np.array([0.1 ...]), ...}

    Parameters
    ----------

    scores : list of dict
        List of dicts of the scores for all scorers. This is a flat list,
        assumed originally to be of row major order.

    Example
    -------

    >>> scores = [{'a': 1, 'b':10}, {'a': 2, 'b':2}, {'a': 3, 'b':3},
    ...           {'a': 10, 'b': 10}]                         # doctest: +SKIP
    >>> _aggregate_score_dicts(scores)                        # doctest: +SKIP
    {'a': array([1, 2, 3, 10]),
     'b': array([10, 2, 3, 10])}
    """
    return {
        key: np.asarray([score[key] for score in scores])
        if isinstance(scores[0][key], numbers.Number)
        else [score[key] for score in scores]
        for key in scores[0]
    }
