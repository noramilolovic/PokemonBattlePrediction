
from datetime import date
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

from src.pipeline import get_train
from src.models import get_models

X, y = get_train()
print("Shape:", X.shape)

cv = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=3,
    random_state=42
)

LABEL = "v3 team+timeline+species+types (1057 features)"

lines = []

for name, model in get_models().items():
    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1
    )

    line = (
        f"| {date.today()} | {LABEL} | {name} | "
        f"{scores.mean():.4f} | {scores.std():.4f} |"
    )

    print(line)
    lines.append(line)

with open("experiments.md", "a", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
