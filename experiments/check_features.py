from src.data import load_train
from src.features_team import build_team_frame

battles = load_train()
X = build_team_frame(battles)

# 1. Shape: one row per battle, many feature columns
print("shape:", X.shape)  # expect (10000, ~50)

# 2. No missing values
print("NaN count:", int(X.isna().sum().sum()))  # expect 0

# 3. Index matches the battles, no duplicates
print("unique ids:", X.index.is_unique, len(X.index))

# 4. Constant columns carry no information
const = [c for c in X.columns if X[c].nunique() <= 1]
print("constant columns:", const)

# 5. Sanity check against the first battle from the starter notebook:
# P1 speeds are 61,100,45,65,120,115 and the P2 lead speed is 40
row = X.iloc[0]
print("p1_mean_spe:", round(row["p1_mean_spe"], 2))          # expect 84.33
print("p2_lead_spe:", row["p2_lead_spe"])                    # expect 40
print("p1_n_faster_than_lead:", row["p1_n_faster_than_lead"])  # expect 6
print("p1_max_spe_minus_lead:", row["p1_max_spe_minus_lead"])  # expect 80

# 6. Ranges look reasonable
print(X.describe().T[["min", "mean", "max"]].round(1))

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from src.data import get_target

y = get_target(battles)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)

starter = ["p1_mean_hp", "p1_mean_spe", "p1_mean_atk", "p1_mean_def",
           "p2_lead_hp", "p2_lead_spe"]

acc_all = cross_val_score(rf, X, y, cv=skf, scoring="accuracy")
acc_starter = cross_val_score(rf, X[starter], y, cv=skf, scoring="accuracy")

print("all features:", acc_all.mean().round(4), "+/-", acc_all.std().round(4))
print("starter only:", acc_starter.mean().round(4), "+/-", acc_starter.std().round(4))
print("baseline (always majority class):", max(y.mean(), 1 - y.mean()).round(4))
