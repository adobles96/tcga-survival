# Predicting Survival on TCGA-LUAD using Pathology Foundation Models and Genomics Data

## Best Hparams

### Demo only
```
model = GradientBoostingSurvivalAnalysis(
    learning_rate=0.1,
    n_estimators=200,
    subsample=0.8,
    min_samples_leaf=4,
    max_features=0.5,
    # validation_fraction=0.2,
    # n_iter_no_change=5,
    dropout_rate=0.1,
    random_state=SEED
)
```

Actual best:
```
    learning_rate=1,
    n_estimators=200,
    subsample=0.8,
    min_samples_leaf=4,
    max_features=0.5,
    # validation_fraction=0.2,
    # n_iter_no_change=5,
    dropout_rate=0.1,
    random_state=SEED
```

### WSI only
uses 5 components
```
    learning_rate=1,
    n_estimators=200,
    subsample=1,
    min_samples_leaf=8,
    max_features=0.8,
    max_depth=3,
    # validation_fraction=0.2,
    # n_iter_no_change=5,
    dropout_rate=0.2,
    random_state=SEED
```


### Omics only
```
model = GradientBoostingSurvivalAnalysis(
    learning_rate=0.01,
    n_estimators=200,
    subsample=0.6,
    min_samples_leaf=6,
    max_features=0.5,
    # validation_fraction=0.2,
    # n_iter_no_change=5,
    dropout_rate=0.3,
    random_state=SEED
)
```

### Demo + WSI
Uses 5 components for wsi
```
    learning_rate=1,
    n_estimators=200,
    subsample=0.9,
    min_samples_leaf=8,
    max_features=0.5,
    max_depth=2,
    # validation_fraction=0.2,
    # n_iter_no_change=5,
    dropout_rate=0.2,
    random_state=SEED
```


### Demo + Omics
Uses 5 components for omics
```
    learning_rate=0.1,
    n_estimators=200,
    subsample=0.8,
    min_samples_leaf=4,
    max_features=0.2,
    max_depth=3,
    # validation_fraction=0.2,
    # n_iter_no_change=5,
    dropout_rate=0.2,
    random_state=SEED
```

### WSI + Omics
5 components for both
```
    learning_rate=1,
    n_estimators=200,
    subsample=1,
    min_samples_leaf=8,
    max_features=0.5,
    max_depth=2,
    # validation_fraction=0.2,
    # n_iter_no_change=5,
    dropout_rate=0.2,
    random_state=SEED
```

### All
5 components for both
```
    learning_rate=5,
    n_estimators=200,
    subsample=1,
    min_samples_leaf=8,
    max_features=0.5,
    max_depth=2,
    # validation_fraction=0.2,
    # n_iter_no_change=5,
    dropout_rate=0.1,
    random_state=SEED
```