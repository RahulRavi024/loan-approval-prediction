
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import  Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

df = pd.read_csv("data/train.csv")
df = df.dropna(subset=["Loan_Status"])

X = df.drop(columns=["Loan_ID", "Loan_Status"])
y = df["Loan_Status"].map({"N": 0, "Y": 1})

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numerical_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numerical_pipeline, numerical_cols),
    ("cat", categorical_pipeline, categorical_cols)
])

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42)
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = {}

for name, classifier in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

    scores = cross_val_score(
        pipeline, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1
    )

    results[name] = scores.mean()

    print(f"\n{name}")
    print("Mean CV F1:", round(scores.mean(), 4))
    print("CV F1 standard deviation:", round(scores.std(), 4))

best_model_name = max(results, key=results.get)

print("\nBest baseline model:", best_model_name)

param_grids = {
    "Logistic Regression": {
        "classifier__C": [0.1, 1, 10],
        "classifier__class_weight": [None, "balanced"]
    },
    "Decision Tree": {
        "classifier__max_depth": [3, 5, 10, None],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__class_weight": [None, "balanced"]
    },
    "Random Forest": {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [5, 10, None],
        "classifier__min_samples_split": [2, 5],
        "classifier__class_weight": [None, "balanced"]
    }
}

best_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", models[best_model_name])
])

search = GridSearchCV(
    best_pipeline,
    param_grids[best_model_name],
    cv=cv,
    scoring="f1",
    n_jobs=-1
)

search.fit(X_train, y_train)

best_pipeline = search.best_estimator_
predictions = best_pipeline.predict(X_test)

print("\nBest model:", best_model_name)
print("Best parameters:", search.best_params_)
print("Best CV F1:", round(search.best_score_, 4))

print("\nTest Results")
print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
print("Precision:", round(precision_score(y_test, predictions, zero_division=0), 4))
print("Recall:", round(recall_score(y_test, predictions, zero_division=0), 4))
print("F1 Score:", round(f1_score(y_test, predictions, zero_division=0), 4))

print("\nClassification Report")
print(classification_report(
    y_test, predictions,
    target_names=["Rejected", "Approved"],
    zero_division=0
))

print("Confusion Matrix")
print(confusion_matrix(y_test, predictions))

joblib.dump(best_pipeline, "model.pkl")

print("\nModel saved successfully as model.pkl")
