
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier

from src.pipeline import get_train

# Load our V3 features
X, y = get_train()

print("Dataset shape:", X.shape)

# Create the model
model = make_pipeline(
    SimpleImputer(strategy="median"),
    HistGradientBoostingClassifier(random_state=42)
)

# Settings we want to test
parameters = {
    "histgradientboostingclassifier__max_iter": [100, 200, 300, 400],
    "histgradientboostingclassifier__learning_rate": [0.03, 0.05, 0.1],
    "histgradientboostingclassifier__max_leaf_nodes": [7, 15, 31],
    "histgradientboostingclassifier__l2_regularization": [0.1, 1.0, 5.0]
}

# Use 5-fold cross-validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

search = RandomizedSearchCV(
    estimator=model,
    param_distributions=parameters,
    n_iter=12,
    scoring="accuracy",
    cv=cv,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

print("Starting hyperparameter tuning...")

search.fit(X, y)

print("\nBest accuracy:", round(search.best_score_, 4))
print("\nBest parameters:")

for name, value in search.best_params_.items():
    print(name, "=", value)
