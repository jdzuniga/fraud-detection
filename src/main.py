import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import RandomizedSearchCV
from imblearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, recall_score, accuracy_score, confusion_matrix, classification_report
from xgboost import XGBClassifier
from sklearn.metrics import precision_recall_curve


dtypes = {"Time": np.uint32,
          "Class": np.bool}

data = pd.read_csv("../data/creditcard.csv", dtype=dtypes)

data["isFraud"] = data["Class"]
data.drop("Class", axis=1, inplace=True)

data = data.drop_duplicates(keep="first")


X = data.drop(["isFraud"], axis=1)
y = data["isFraud"]

train_size = int(0.8 * len(data))
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]

train_fraud_percentage = y_train.value_counts(normalize=True) * 100
test_fraud_percentage = y_test.value_counts(normalize=True) * 100

print("Not Frauds: {:.2f}% of the train set".format(train_fraud_percentage.iloc[0]))
print("Frauds: {:.2f}% of the train set".format(train_fraud_percentage.iloc[1]),'\n')

print("Not Frauds: {:.2f}% of the test set".format(test_fraud_percentage.iloc[0]))
print("Frauds: {:.2f}% of the test set".format(test_fraud_percentage.iloc[1]), '\n')



scv = StratifiedKFold(n_splits=3, shuffle=True, random_state=11)

for fold, (train_idx, val_idx) in enumerate(scv.split(X_train, y_train)):
    X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
    X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]

    _, train_count = np.unique(y_tr, return_counts=True)
    _, valid_count = np.unique(y_val, return_counts=True)
    print(f"fold {fold+1} -", f"train: {train_count / sum(train_count)}", f"valid: {valid_count / sum(valid_count)}")


other_features = X.drop(["Time", "Amount"], axis=1).columns.to_list()

model = XGBClassifier(
    objective='binary:logistic',
    eval_metric='aucpr',
    use_label_encoder=False,
    random_state=11,
    n_jobs=-1,
)

preprocessor = ColumnTransformer([
    ("time_preprocess", RobustScaler(), ["Time"]),
    ("amount_preprocess", StandardScaler(), ["Amount"]),
    ('pass', 'passthrough', other_features)
    ]
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("smote", SMOTE(random_state=11)),
    ("clf", model)
])

param_dist = {
    'clf__n_estimators': [100, 300, 500, 800, 1000],
    'clf__learning_rate': [0.01, 0.05, 0.1, 0.2],
    'clf__max_depth': [3, 5, 7, 9, 11],
    'clf__subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'clf__colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'clf__gamma': [0, 0.1, 0.3, 0.5, 1],
    'clf__min_child_weight': [1, 3, 5, 7, 10],
    'clf__reg_alpha': [0, 0.01, 0.1, 1, 10],
    'clf__reg_lambda': [0.1, 1, 5, 10, 20]
}

scoring = {
    'accuracy': make_scorer(accuracy_score),
    'recall': make_scorer(recall_score, pos_label=1),
    'roc_auc': 'roc_auc'
}

search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_dist,
    n_iter=5,
    cv=scv,
    scoring=scoring,
    refit='recall',
    verbose=2,
    n_jobs=-1,
    random_state=11
)

search.fit(X_train, y_train)

print("Best CV Recall:", search.cv_results_['mean_test_recall'][search.best_index_])
print("Best CV Accuracy:", search.cv_results_['mean_test_accuracy'][search.best_index_])
print("Best CV ROC-AUC:", search.cv_results_['mean_test_roc_auc'][search.best_index_])


best_model = search.best_estimator_
y_pred_prob = best_model.predict_proba(X_test)[:, 1]

precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_prob)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls)

best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]
print(f"Best threshold (by F1): {best_threshold:.3f}")
print(f"Precision: {precisions[best_idx]:.3f}, Recall: {recalls[best_idx]:.3f}")

y_pred_opt = (y_pred_prob >= best_threshold).astype(int)
print(confusion_matrix(y_test, y_pred_opt))
print(classification_report(y_test, y_pred_opt, digits=3))