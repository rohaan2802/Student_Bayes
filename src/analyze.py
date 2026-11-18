"""Reproducible baseline evaluation and portable prediction for Student_Bayes.

Predictive models fit all preprocessing inside the training split or CV fold.
The supplied notebook contains the separate Bayesian analysis.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import platform
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             log_loss, mean_absolute_error, mean_squared_error, r2_score)
from sklearn.model_selection import (RepeatedStratifiedKFold, StratifiedKFold,
                                     cross_val_score, train_test_split)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ['study_hours', 'sleep_hours', 'social_time']
CLASSES = ['bad', 'good', 'excellent']
SEED = 42


def validate_data(frame):
    """Reject malformed inputs before fitting, with actionable errors."""
    required = FEATURES + ['quiz_score', 'student_type']
    if len(frame.columns) != 5 or set(frame.columns) != set(required):
        raise ValueError('Expected exactly: ' + ', '.join(required))
    if len(frame) < 30 or frame.isna().any().any():
        raise ValueError('Data must contain at least 30 complete rows')
    if set(frame.student_type) != set(CLASSES):
        raise ValueError('Expected bad, good, excellent classes')
    if frame.student_type.value_counts().min() < 7:
        raise ValueError('Each class needs at least seven rows for training-only cross-validation')
    if not all(pd.api.types.is_numeric_dtype(frame[c]) for c in FEATURES + ['quiz_score']):
        raise ValueError('Predictors and quiz_score must have numeric types')
    try:
        values = frame[FEATURES + ['quiz_score']].to_numpy(dtype=float)
    except (ValueError, TypeError) as error:
        raise ValueError('Predictors and quiz_score must be numeric') from error
    if not np.isfinite(values).all():
        raise ValueError('Numeric values must be finite')
    if (frame[FEATURES].std() == 0).any():
        raise ValueError('Predictors must vary')
    if (frame[FEATURES] < 0).any().any():
        raise ValueError('Recorded hours cannot be negative')
    return frame


def load_data(path):
    return validate_data(pd.read_csv(path))


def make_classifier():
    """Scaling is fitted separately inside each training split/fold."""
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))


def split_rows(frame, seed=SEED):
    return train_test_split(np.arange(len(frame)), test_size=.2,
                            stratify=frame.student_type, random_state=seed)


def wilson_interval(successes, total, confidence=.95):
    """Binomial Wilson interval expressing uncertainty in a small test set."""
    if total <= 0 or not 0 <= successes <= total:
        raise ValueError('Require 0 <= successes <= total and total > 0')
    z = norm.ppf((1 + confidence) / 2)
    p = successes / total
    center = (p + z*z/(2*total)) / (1 + z*z/total)
    radius = z*np.sqrt(p*(1-p)/total + z*z/(4*total*total))/(1+z*z/total)
    return [float(max(0, center-radius)), float(min(1, center+radius))]


def analyze(frame, seed=SEED):
    """Return fixed-holdout metrics and row-level classification probabilities."""
    validate_data(frame)
    train, test = split_rows(frame, seed)
    model = make_classifier().fit(frame.iloc[train][FEATURES], frame.iloc[train].student_type)
    pred = model.predict(frame.iloc[test][FEATURES])
    prob = model.predict_proba(frame.iloc[test][FEATURES])
    if not np.isfinite(prob).all() or not np.allclose(prob.sum(axis=1), 1):
        raise RuntimeError('Classifier returned invalid probabilities')
    actual = frame.iloc[test].student_type.to_numpy()
    dummy = DummyClassifier(strategy='prior').fit(frame.iloc[train][FEATURES], frame.iloc[train].student_type)
    regression = LinearRegression().fit(frame[FEATURES], frame.quiz_score)
    one_hot = np.column_stack([actual == c for c in model.classes_])
    result = dict(rows=len(frame), missing=int(frame.isna().sum().sum()),
                  duplicates=int(frame.duplicated().sum()), seed=seed,
                  train_rows=len(train), test_rows=len(test),
                  baseline_accuracy=accuracy_score(actual, pred),
                  accuracy_wilson_95=wilson_interval(int((actual == pred).sum()), len(test)),
                  baseline_log_loss=log_loss(actual, prob, labels=model.classes_),
                  multiclass_brier=float(np.mean(np.sum((prob-one_hot)**2, axis=1))),
                  dummy_accuracy=accuracy_score(actual, dummy.predict(frame.iloc[test][FEATURES])),
                  baseline_confusion=confusion_matrix(actual, pred, labels=CLASSES).tolist(),
                  confusion_labels=CLASSES,
                  baseline_report=classification_report(actual, pred, output_dict=True, zero_division=0),
                  descriptive_ols_r2=regression.score(frame[FEATURES], frame.quiz_score),
                  ols_coefficients=dict(zip(FEATURES, regression.coef_)),
                  correlation=frame[FEATURES+['quiz_score']].corr().to_dict())
    predictions = pd.DataFrame(prob, columns=model.classes_)
    predictions.insert(0, 'source_row', test)
    predictions['actual'] = actual
    predictions['predicted'] = pred
    predictions['confidence'] = prob.max(axis=1)
    return result, predictions


def evaluate(frame, seed=SEED):
    """Training-only CV, separate holdout regression, and train-fold importance.

    Repeated CV never consumes the reserved holdout. Permutation importance
    uses validation folds within the training split. Correlated predictors can
    share importance; importance is not a causal effect.
    """
    train, test = split_rows(frame, seed)
    xtrain, xtest = frame.iloc[train][FEATURES], frame.iloc[test][FEATURES]
    ytrain = frame.iloc[train].student_type
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=seed)
    scores = cross_val_score(make_classifier(), xtrain, ytrain, cv=cv, scoring='accuracy')
    regression_rows = []
    for name, estimator in [('Mean-only', DummyRegressor()),
                            ('Study-only OLS', LinearRegression()),
                            ('Multiple OLS', LinearRegression()),
                            ('Ridge (alpha=1)', make_pipeline(StandardScaler(), Ridge(alpha=1)))]:
        columns = ['study_hours'] if name == 'Study-only OLS' else FEATURES
        estimator.fit(xtrain[columns], frame.iloc[train].quiz_score)
        pred = estimator.predict(xtest[columns]); truth = frame.iloc[test].quiz_score
        regression_rows.append(dict(model=name, rmse=float(np.sqrt(mean_squared_error(truth,pred))),
                                    mae=mean_absolute_error(truth,pred), r2=r2_score(truth,pred)))
    drops = []
    for fold, (fit_idx, valid_idx) in enumerate(StratifiedKFold(5,shuffle=True,random_state=seed).split(xtrain,ytrain)):
        model = make_classifier().fit(xtrain.iloc[fit_idx],ytrain.iloc[fit_idx])
        importance = permutation_importance(model,xtrain.iloc[valid_idx],ytrain.iloc[valid_idx],
                                           scoring='neg_log_loss',n_repeats=10,random_state=seed+fold)
        drops.append(importance.importances)
    drops = np.concatenate(drops,axis=1)
    importance = pd.DataFrame(dict(feature=FEATURES,mean_log_loss_increase=drops.mean(axis=1),
                                   std_log_loss_increase=drops.std(axis=1)))
    evaluation = dict(cv_scope='classification training partition only',cv_folds=5,cv_repeats=3,
                      cv_accuracy_mean=float(scores.mean()),cv_accuracy_sd=float(scores.std(ddof=1)),
                      cv_accuracy_min=float(scores.min()),cv_accuracy_max=float(scores.max()),
                      cv_scores=scores.tolist(),regression_holdout=regression_rows)
    return evaluation, importance


def portable_model(frame, seed=SEED):
    """Export numeric parameters without pickle, using the fixed training split."""
    train, _ = split_rows(frame, seed)
    pipeline = make_classifier().fit(frame.iloc[train][FEATURES],frame.iloc[train].student_type)
    scale, logistic = pipeline.steps[0][1], pipeline.steps[1][1]
    return dict(model='frequentist multinomial logistic regression',seed=seed,features=FEATURES,
                classes=logistic.classes_.tolist(),mean=scale.mean_.tolist(),scale=scale.scale_.tolist(),
                coefficients=logistic.coef_.tolist(),intercept=logistic.intercept_.tolist(),
                observed_min=frame.iloc[train][FEATURES].min().tolist(),
                observed_max=frame.iloc[train][FEATURES].max().tolist())


def predict_parameters(model, values):
    values = np.asarray(values,dtype=float)
    if values.shape != (3,) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Provide three finite, nonnegative habit values')
    scores = np.asarray(model['coefficients']) @ ((values-model['mean'])/model['scale']) + model['intercept']
    exp = np.exp(scores-scores.max()); probabilities = exp/exp.sum()
    return dict(zip(model['classes'],probabilities.tolist()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=ROOT/'data/Syn_student.csv')
    parser.add_argument('--output',type=Path,default=ROOT/'reports')
    parser.add_argument('--seed',type=int,default=SEED)
    parser.add_argument('--predict',type=float,nargs=3,metavar=('STUDY','SLEEP','SOCIAL'),
                        help='Predict probabilities with the training-partition baseline')
    args = parser.parse_args()
    try:
        frame = load_data(args.data)
        if args.predict is not None:
            print(json.dumps(predict_parameters(portable_model(frame,args.seed),args.predict),indent=2))
            return
        result, predictions = analyze(frame,args.seed)
        evaluation, importance = evaluate(frame,args.seed)
        result.update(evaluation)
        args.output.mkdir(parents=True,exist_ok=True)
        def save(name,value):
            (args.output/name).write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
        save('baseline_metrics.json',result)
        save('demo_model.json',portable_model(frame,args.seed))
        save('baseline_environment.json',dict(python=platform.python_version(),
            packages={n:importlib.metadata.version(n) for n in ['numpy','pandas','scikit-learn','scipy','matplotlib','Pillow']},
            dataset_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest()))
        predictions.to_csv(args.output/'baseline_predictions.csv',index=False)
        importance.to_csv(args.output/'feature_importance.csv',index=False)
        pd.DataFrame(result['regression_holdout']).to_csv(args.output/'regression_comparison.csv',index=False)
        frame.groupby('student_type')[FEATURES+['quiz_score']].agg(['mean','std','median']).to_csv(args.output/'group_summary.csv')
        print(f"Student_Bayes | {len(frame)} rows | holdout accuracy {result['baseline_accuracy']:.1%}")
        print(f"Training CV: {result['cv_accuracy_mean']:.1%} +/- {result['cv_accuracy_sd']:.1%}; holdout log loss: {result['baseline_log_loss']:.4f}")
        print(f'Reports: {args.output.resolve()}')
    except (ValueError,OSError) as error:
        parser.exit(2,f'Input error: {error}\n')


if __name__ == '__main__':
    main()
