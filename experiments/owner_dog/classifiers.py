"""Classical pair classifiers with explicit, class-balanced weight conventions."""

import warnings

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC


def candidate_grid():
    """Return fresh small grids to select only inside each outer training fold."""
    return {
        'logistic': [{'lambda': .01}, {'lambda': .1}],
        'linear_svm': [{'lambda': .01}, {'lambda': .1}],
        'rbf_svm': [{'C': 1000., 'gamma': 'scale'}, {'C': 10000., 'gamma': 'scale'}],
        'shrinkage_lda': [{'shrinkage': 'auto'}, {'shrinkage': .5}],
        'extra_trees': [{'max_depth': 4}, {'max_depth': 8}],
        'hist_gbdt': [{'max_leaf_nodes': 7}, {'max_leaf_nodes': 15}],
    }


def describe_heads():
    return {
        'implementation': 'classical_heads_photo_sweep_20261102.py',
        'deep_learning_used': False,
        'reinforcement_learning_used': False,
        'candidate_grid': candidate_grid(),
        'input_weights': 'Strictly positive weights sum to 1; each class contributes 0.5',
        'scaling': 'StandardScaler fitted on training rows with the normalized input weights',
        'logistic': {'solver': 'lbfgs', 'max_iter': 2000, 'tol': 1e-8,
                     'C': '1/lambda with normalized input weights', 'class_weight': None},
        'linear_svm': {'loss': 'squared_hinge', 'dual': False, 'max_iter': 5000,
                       'tol': 1e-8, 'C': '1/lambda with normalized input weights', 'class_weight': None},
        'rbf_svm': {'kernel': 'rbf', 'cache_size_MiB': 128, 'tol': .001,
                    'C_units': 'Normalized input weights summing to 1, not unweighted-row units',
                    'class_weight': None},
        'shrinkage_lda': {'solver': 'lsqr', 'priors': [.5, .5],
                          'fit_sample_weight': 'Unsupported; equal within-class row weights required'},
        'extra_trees': {'n_estimators': 64, 'min_samples_leaf': 10, 'max_features': .7,
                        'bootstrap': False, 'n_jobs': 1,
                        'fit_weights': 'Input weights multiplied by training row count'},
        'hist_gbdt': {'max_iter': 60, 'max_bins': 63, 'learning_rate': .08, 'min_samples_leaf': 15,
                      'l2_regularization': 1., 'early_stopping': False,
                      'fit_weights': 'Input weights multiplied by training row count for leaf L2 units'},
        'score_meaning': 'Decision function or class-1 probability used only as a retrieval ranking score',
        'selection_limit': 'Two fixed candidates per head; not an exhaustive model optimization',
    }


def _validate(x, y, weights):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y)
    weights = np.asarray(weights, dtype=np.float64)
    if x.ndim != 2 or not x.shape[1] or len(x) < 4 or not np.isfinite(x).all():
        raise ValueError('Expected a finite training feature matrix')
    if y.shape != (len(x),) or weights.shape != (len(x),):
        raise ValueError('Labels and weights must match training rows')
    if set(np.unique(y)) != {0, 1}:
        raise ValueError('Both positive and negative binary labels are required')
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError('Filter forbidden rows before fitting; retained weights must be positive')
    if not np.isclose(weights.sum(), 1., atol=1e-10):
        raise ValueError('Weights must sum to 1')
    if any(not np.isclose(weights[y == value].sum(), .5, atol=1e-10) for value in [0, 1]):
        raise ValueError('Each class must contribute half of the weight')
    return x, y.astype(np.int64), weights


def fit_head(x, y, weights, head_name, params, seed):
    """Fit training-only scaling and a classical classifier, retaining its state."""
    x, y, weights = _validate(x, y, weights)
    if head_name not in candidate_grid():
        raise ValueError(f'Unsupported classical head: {head_name}')
    scaler = StandardScaler().fit(x, sample_weight=weights)
    z = scaler.transform(x)
    fit_weights = weights
    if head_name == 'logistic':
        classifier = LogisticRegression(C=1/float(params['lambda']), solver='lbfgs',
                                        max_iter=2000, tol=1e-8)
    elif head_name == 'linear_svm':
        classifier = LinearSVC(C=1/float(params['lambda']), dual=False,
                               max_iter=5000, tol=1e-8)
    elif head_name == 'rbf_svm':
        classifier = SVC(C=float(params['C']), gamma=params['gamma'], cache_size=128)
    elif head_name == 'shrinkage_lda':
        for value in [0, 1]:
            selected = weights[y == value]
            if not np.allclose(selected, selected[0], atol=1e-12, rtol=1e-10):
                raise ValueError('LDA without sample_weight requires constant weights within each class')
        classifier = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=params['shrinkage'],
                                               priors=[.5, .5])
    elif head_name == 'extra_trees':
        classifier = ExtraTreesClassifier(n_estimators=64, max_depth=int(params['max_depth']),
            min_samples_leaf=10, max_features=.7, bootstrap=False, n_jobs=1, random_state=int(seed))
        fit_weights = weights * len(y)
    else:
        classifier = HistGradientBoostingClassifier(max_leaf_nodes=int(params['max_leaf_nodes']),
            max_iter=60, max_bins=63, learning_rate=.08, min_samples_leaf=15, l2_regularization=1.,
            early_stopping=False, random_state=int(seed))
        fit_weights = weights * len(y)
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        if head_name == 'shrinkage_lda':
            classifier.fit(z, y)
        else:
            classifier.fit(z, y, sample_weight=fit_weights)
    diagnostics = {
        'training_rows': int(len(y)), 'positive_rows': int(y.sum()),
        'negative_rows': int((y == 0).sum()), 'input_weight_sum': float(weights.sum()),
        'positive_weight_sum': float(weights[y == 1].sum()),
        'fit_weight_sum': None if head_name == 'shrinkage_lda' else float(fit_weights.sum()),
        'lda_uses_equal_class_priors': head_name == 'shrinkage_lda',
        'seed': int(seed), 'classes': classifier.classes_.tolist(),
    }
    if hasattr(classifier, 'n_iter_'):
        diagnostics['iterations'] = np.asarray(classifier.n_iter_).tolist()
    if hasattr(classifier, 'n_support_'):
        diagnostics['support_rows_per_class'] = classifier.n_support_.tolist()
    return dict(scaler=scaler, classifier=classifier, name=head_name, params=dict(params),
                fit_diagnostics=diagnostics)


def score_head(model, x):
    """Return class-1-oriented ranking scores, without a probability guarantee."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError('Expected a finite scoring matrix')
    z = model['scaler'].transform(x)
    classifier = model['classifier']
    if hasattr(classifier, 'decision_function'):
        scores = classifier.decision_function(z)
    else:
        positive = int(np.flatnonzero(classifier.classes_ == 1)[0])
        scores = classifier.predict_proba(z)[:, positive]
    scores = np.asarray(scores, dtype=float)
    if scores.shape != (len(x),) or not np.isfinite(scores).all():
        raise RuntimeError('Classifier returned invalid ranking scores')
    return scores
