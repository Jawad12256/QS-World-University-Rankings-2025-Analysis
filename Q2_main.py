#Section 1 - Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Section 2 - Load Data + Clean Column Names

df = pd.read_csv("data/QS_DATASET.csv", encoding="latin1")
print(df.head())
print(df.shape)
df.columns = df.columns.str.strip()

# Section 3 - Create Target Variable

df["RANK_2025"] = pd.to_numeric(df["RANK_2025"], errors="coerce")
df = df.dropna(subset=["RANK_2025"])

df["Top100"] = (df["RANK_2025"] <= 100).astype(int)

y = df["Top100"]

# Section 4 - Select Features

features = [
    "Academic_Reputation_Score",
    "Employer_Reputation_Score",
    "Faculty_Student_Score",
    "Citations_per_Faculty_Score",
    "International_Faculty_Score",
    "International_Students_Score",
    "International_Research_Network_Score",
    "Employment_Outcomes_Score",
    "Sustainability_Score"
]

X = df[features]

# Section 5 - Handle Missing Values

X = X.apply(pd.to_numeric, errors="coerce")
X = X.fillna(X.median())

# Section 6 - Train/Validation/Test Split

from sklearn.model_selection import train_test_split

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.4,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.5,
    random_state=42,
    stratify=y_temp
)

# Section 7 - Forward Feature Selection (VALIDATION-BASED)

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

remaining_features = list(features)
selected_features = []

train_scores = []
val_scores = []

for i in range(len(features)):

    best_feature = None
    best_score = -np.inf
    best_scaler = None
    best_model = None

    for f in remaining_features:

        trial_features = selected_features + [f]

        scaler = StandardScaler()

        X_train_s = scaler.fit_transform(X_train[trial_features])
        X_val_s = scaler.transform(X_val[trial_features])

        model = LogisticRegression(
            class_weight="balanced",
            max_iter=5000
        )

        model.fit(X_train_s, y_train)

        val_prob = model.predict_proba(X_val_s)[:, 1]
        score = roc_auc_score(y_val, val_prob)

        if score > best_score:
            best_score = score
            best_feature = f
            best_scaler = scaler
            best_model = model

    selected_features.append(best_feature)
    remaining_features.remove(best_feature)

    X_train_best = best_scaler.transform(X_train[selected_features])

    train_scores.append(
        accuracy_score(y_train, best_model.predict(X_train_best))
    )

    val_scores.append(best_score)

    print(f"{i+1} features | added: {best_feature} | val={best_score:.3f}")

# Section 8 - Choosing Best Feature Set

best_idx = np.argmax(val_scores)

best_features = selected_features[:best_idx + 1]

print("Best number of features:", best_idx + 1)
print("Best features:", best_features)


# Section 9 - Final Model Training

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

scaler_final = StandardScaler()

X_train_final = scaler_final.fit_transform(X_train[best_features])
X_test_final = scaler_final.transform(X_test[best_features])

final_model = LogisticRegression(
    class_weight="balanced",
    max_iter=5000
)

final_model.fit(X_train_final, y_train)

y_pred = final_model.predict(X_test_final)
y_prob = final_model.predict_proba(X_test_final)[:, 1]

# Section 10 - Model Complexity Plot

plt.figure()

plt.plot(
    range(1, len(train_scores) + 1),
    train_scores,
    marker="o",
    label="Train Accuracy"
)

plt.plot(
    range(1, len(val_scores) + 1),
    val_scores,
    marker="o",
    label="Validation Accuracy"
)

plt.axvline(best_idx + 1, linestyle="--", color="red", label="Optimal Complexity")

plt.xlabel("Number of Features")
plt.ylabel("Accuracy")
plt.title("Model Complexity vs Accuracy")

plt.legend()
plt.grid()
plt.show()

# Section 11 - Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

#Section 12 - Metrics

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))

# Section 13 - ROC-AUC Curve

from sklearn.metrics import roc_curve

fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], "--")

plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.show()

# Section 14 - Feature Coefficients (FIXED ALIGNMENT)

coef = final_model.coef_[0]

coef_df = pd.DataFrame({
    "Feature": best_features,
    "Coefficient": coef
})

coef_df = coef_df.sort_values(by="Coefficient", ascending=False)

print(coef_df)

coef_df.plot(kind="barh", x="Feature", y="Coefficient")

plt.title("Logistic Regression Coefficients")
plt.xlabel("Coefficient Value")
plt.grid()
plt.show()

# Section 15 - Logistic Curve (NO DATA LEAKAGE)

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

coef_importance = np.abs(final_model.coef_[0])
feature_name = best_features[np.argmax(coef_importance)]

# use training data only
X_vis = X_train[[feature_name]].copy()

scaler_vis = StandardScaler()
X_vis_scaled = scaler_vis.fit_transform(X_vis)

vis_model = LogisticRegression(
    class_weight="balanced",
    max_iter=5000
)

vis_model.fit(X_vis_scaled, y_train)

X_range = np.linspace(
    X_vis_scaled.min(),
    X_vis_scaled.max(),
    300
).reshape(-1, 1)

y_curve = vis_model.predict_proba(X_range)[:, 1]

plt.figure()
plt.plot(X_range, y_curve)
plt.title(f"Logistic Curve: {feature_name}")
plt.xlabel("Standardised Feature")
plt.ylabel("Probability of Top 100")
plt.grid()
plt.show()