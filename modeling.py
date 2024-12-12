import numpy as np
import optuna
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.metrics import integrated_brier_score, cumulative_dynamic_auc
from data_utils import load_dataset


def get_gbdt_hparams(trial):
    return {
        'loss': 'coxph',
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 50, 300, step=25),
        'subsample': trial.suggest_float('subsample', 0.3, 1.0, step=0.1),
        'max_depth': trial.suggest_int('max_depth', 2, 10),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 8),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 8),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'n_iter_no_change': (
            trial.suggest_int('n_iter_no_change', 5, 20)
            if trial.suggest_categorical('early_stopping', [True, False]) else None
        ),
        'dropout_rate': trial.suggest_float('dropout_rate', 0.0, 0.7, step=0.1),
    }


def cindex_score_fn(model, X, y):
    return model.score(X, y)


def run_trial(X, y, trial, model_type=GradientBoostingSurvivalAnalysis, random_state=42):
    if model_type == GradientBoostingSurvivalAnalysis:
        model = model_type(**get_gbdt_hparams(trial), random_state=random_state)
    return cross_val_score(model, X, y, cv=5, scoring=cindex_score_fn).mean()


def hparam_tune(X, y, model_type, random_state=42, num_trials=20):
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: run_trial(X, y, trial, model_type, random_state=random_state),
        n_trials=num_trials
    )
    print(f'Best params: {study.best_params}')
    print('Training final model on all data')
    params = study.best_params
    params.pop('early_stopping')  # avoids error in model instantiation
    return model_type(**params, random_state=random_state).fit(X, y)


class CustomStratifiedKFold(StratifiedKFold):
    """ Accounts for 2-dimensional labels in survival data. """
    def split(self, X, y, groups=None):
        return super().split(X, y['event'], groups)


def get_test_scores(model, X_test, y_test, y):
    cindex = model.score(X_test, y_test)
    times = np.percentile(y['survival_time'], np.linspace(10, 81, 20))
    preds = np.asarray([[fn(t) for t in times] for fn in model.predict_survival_function(X_test)])
    bscore = integrated_brier_score(y, y_test, preds, times)
    risk_scores = model.predict(X_test)
    aucs = cumulative_dynamic_auc(y, y_test, risk_scores, times)
    return cindex, bscore, aucs


def tune_and_eval(model_type=GradientBoostingSurvivalAnalysis, random_state=42, include_demo=True,
                  include_wsi=False, include_omics=False):
    X_demo, X_wsi, X_omics, y = load_dataset(include_demo=include_demo, include_wsi=include_wsi,
                                             include_omics=include_omics)
    X = np.concatenate(list(filter(lambda x: x is not None, [X_demo, X_wsi, X_omics])), axis=1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y['event']
    )
    model = hparam_tune(X_train, y_train, model_type, random_state=random_state)
    cindex, bscore, aucs = get_test_scores(model, X_test, y_test, y)
    print(f'Final model C-index: {cindex:0.3f}')
    print(f'Final model Brier Score: {bscore:0.3f}')
    print(f'Avg AUC: {aucs[-1]:.3f}')
    return cindex, bscore, aucs[-1]


def data_ablation(inlcude_demo=True, include_wsi=True, include_omics=True):
    cinds = []
    bscores = []
    np.random.seed(42)
    for s in np.random.randint(0, 1000, 5):
        c, b, _ = tune_and_eval(
            random_state=s, include_demo=inlcude_demo, include_wsi=include_wsi,
            include_omics=include_omics
        )
        cinds.append(c)
        bscores.append(b)
    print()
    print('-' * 80)
    print('-' * 80)
    print()
    print(f'Test C-index: {np.mean(cinds):.3f} +/- {np.std(cinds):.3f}')
    print(f'Test Brier Score: {np.mean(bscores):.3f} +/- {np.std(bscores):.3f}')


if __name__ == '__main__':
    data_ablation(inlcude_demo=False, include_wsi=True, include_omics=False)
