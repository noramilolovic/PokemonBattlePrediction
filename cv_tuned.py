
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    cross_val_score
)
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier

from src.pipeline import get_train

# Load V3 dataset
X, y = get_train()

print("Dataset shape:", X.shape)

# Tuned model
model = make_pipeline(
    SimpleImputer(strategy="median"),
    HistGradientBoostingClassifier(
        max_iter=400,
        learning_rate=0.05,
        max_leaf_nodes=31,
        l2_regularization=5.0,
        random_state=42
    )
)

# Same validation as our previous experiments
cv = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=3,
    random_state=42
)

print("Evaluating tuned model...")

scores = cross_val_score(
    model,
    X,
    y,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1
)

print("\nMean accuracy:", round(scores.mean(), 4))
print("Standard deviation:", round(scores.std(), 4))
print("Individual fold scores:", scores)
