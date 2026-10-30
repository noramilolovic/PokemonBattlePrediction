| 2026-10-29 | v1 team+timeline (253 features) | logreg | 0.6730 | 0.0063 |
| 2026-10-29 | v1 team+timeline (253 features) | rf | 0.6706 | 0.0076 |
| 2026-10-29 | v1 team+timeline (253 features) | hgb15 | 0.6752 | 0.0047 |
| 2026-10-30 | v2 team+timeline+species (1053 features) | logreg | 0.6575 | 0.0079 |
| 2026-10-30 | v2 team+timeline+species (1053 features) | rf | 0.6706 | 0.0056 |
| 2026-10-30 | v2 team+timeline+species (1053 features) | hgb15 | 0.6771 | 0.0033 |
| 2026-10-30 | v3 team+timeline+species+types (1057 features) | logreg | 0.6571 | 0.0078 |
| 2026-10-30 | v3 team+timeline+species+types (1057 features) | rf | 0.6698 | 0.0076 |
| 2026-10-30 | v3 team+timeline+species+types (1057 features) | hgb15 | 0.6770 | 0.0049 |
| 2026-10-30 | v3 tuned HGB (1057 features) | hgb_tuned | 0.6771 | 0.0063 |

### V3 Hyperparameter Tuning

Model: HistGradientBoostingClassifier

Best parameters:
- max_iter: 400
- learning_rate: 0.05
- max_leaf_nodes: 31
- l2_regularization: 5.0

RandomizedSearchCV best score: 0.6805
Repeated 5-fold CV (3 repeats): 0.6771
Standard deviation: 0.0063

Conclusion:
The tuned V3 model tied the V2 baseline in mean accuracy.
V2 remains the preferred simpler model.