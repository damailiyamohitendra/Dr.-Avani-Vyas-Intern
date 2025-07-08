import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold  # Import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


try:
    df = pd.read_csv('diabetes_prediction_dataset.csv')
    print("Dataset loaded!")
    print(df.head())
    print("\nDataset Info: ")
    df.info()
except FileNotFoundError:
    print("File Not Found!!!")
    exit()
except Exception as e:
    print(f"An error occurred while loading the CSV file: {e}")
    exit()

for col in df.columns:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        print(f"Encoded column '{col}'")

X = df.drop(columns=['diabetes'], axis=1)
y = df['diabetes']
print(f"\nDataset shape: X={X.shape}, y={y.shape} (Classification)")

feature_names = X.columns.tolist()
class_names = [str(c) for c in y.unique()]
X_train_full, X_test_final, y_train_full, y_test_final = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nInitial Full Training data shape: X_train_full={X_train_full.shape}, y_train_full={y_train_full.shape}")
print(f"Final Testing data shape: X_test_final={X_test_final.shape}, y_test_final={y_test_final.shape}")


def _print_evaluation_metrics(model_name, y_true, y_pred, is_final_test=False):
    accuracy = accuracy_score(y_true, y_pred)
    loss = 1 - accuracy
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    if is_final_test:
        print(f"\n--- {model_name} (Final Test Set Evaluation) ---")
    else:
        pass

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Loss (Misclassification Rate): {loss:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)

    return {
        'Accuracy': accuracy,
        'Loss': loss,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Confusion Matrix': cm.tolist()
    }


def _print_average_cv_metrics(model_name, metrics_dict):
    print(f"\n--- {model_name} (Cross-Validation Averages) ---")
    print(f"  Average Accuracy: {metrics_dict['Average Accuracy']:.4f}")
    print(f"  Average Loss: {metrics_dict['Average Loss']:.4f}")
    print(f"  Average Precision: {metrics_dict['Average Precision']:.4f}")
    print(f"  Average Recall: {metrics_dict['Average Recall']:.4f}")
    print(f"  Average F1-Score: {metrics_dict['Average F1-Score']:.4f}")


def run_model_with_kfold(model_instance, model_name, X_data, y_data, n_splits=5, random_state=42):
    print(f"\n--- Starting {model_name} with {n_splits}-Fold Stratified Cross-Validation ---")

    fold_accuracies = []
    fold_precisions = []
    fold_recalls = []
    fold_f1_scores = []

    scaler = StandardScaler()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_num = 1
    for train_index, val_index in skf.split(X_data, y_data):
        print(f"  Fold {fold_num}/{n_splits}...")

        X_train_fold, X_val_fold = X_data.iloc[train_index], X_data.iloc[val_index]
        y_train_fold, y_val_fold = y_data.iloc[train_index], y_data.iloc[val_index]

        X_train_scaled_fold = scaler.fit_transform(X_train_fold)
        X_val_scaled_fold = scaler.transform(X_val_fold)

        model_instance.fit(X_train_scaled_fold, y_train_fold)

        if isinstance(model_instance, LinearRegression):
            y_pred_continuous = model_instance.predict(X_val_scaled_fold)
            y_pred_fold = (y_pred_continuous > 0.5).astype(int)
        else:
            y_pred_fold = model_instance.predict(X_val_scaled_fold)

        fold_accuracies.append(accuracy_score(y_val_fold, y_pred_fold))
        fold_precisions.append(precision_score(y_val_fold, y_pred_fold, zero_division=0))
        fold_recalls.append(recall_score(y_val_fold, y_pred_fold, zero_division=0))
        fold_f1_scores.append(f1_score(y_val_fold, y_pred_fold, zero_division=0))

        fold_num += 1

    avg_accuracy = sum(fold_accuracies) / len(fold_accuracies)
    avg_loss = 1 - avg_accuracy
    avg_precision = sum(fold_precisions) / len(fold_precisions)
    avg_recall = sum(fold_recalls) / len(fold_recalls)
    avg_f1 = sum(fold_f1_scores) / len(fold_f1_scores)

    avg_metrics_results = {
        'Average Accuracy': avg_accuracy,
        'Average Loss': avg_loss,
        'Average Precision': avg_precision,
        'Average Recall': avg_recall,
        'Average F1-Score': avg_f1
    }
    _print_average_cv_metrics(model_name, avg_metrics_results)

    return avg_metrics_results, fold_accuracies


all_model_metrics_cv = {}
model_fold_accuracies = {}

n_splits_for_cv = 10

# K-Nearest Neighbors (KNN)
knn_model = KNeighborsClassifier(n_neighbors=3)
all_model_metrics_cv['KNN'], model_fold_accuracies['KNN'] = run_model_with_kfold(knn_model, 'KNN', X_train_full, y_train_full, n_splits=10)

# Multi-Linear Regression (as a classifier)
lr_model = LinearRegression()
all_model_metrics_cv['Multi-Linear Regression'], model_fold_accuracies['Multi-Linear Regression'] = run_model_with_kfold(lr_model, 'Multi-Linear Regression',
                                                                       X_train_full, y_train_full, n_splits=10)

# Logistic Regression
logreg_model = LogisticRegression(max_iter=1000, random_state=42)
all_model_metrics_cv['Logistic Regression'], model_fold_accuracies['Logistic Regression'] = run_model_with_kfold(
    logreg_model, 'Logistic Regression', X_train_full, y_train_full, n_splits=n_splits_for_cv
)

# SVM
svm_model = SVC(kernel='rbf', random_state=42)
all_model_metrics_cv['SVM'], model_fold_accuracies['SVM'] = run_model_with_kfold(
    svm_model, 'SVM', X_train_full, y_train_full, n_splits=n_splits_for_cv
)

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
all_model_metrics_cv['Random Forest'], model_fold_accuracies['Random Forest'] = run_model_with_kfold(rf_model, 'Random Forest', X_train_full, y_train_full,
                                                             n_splits=10)

rf_model_for_plot = RandomForestClassifier(n_estimators=100, random_state=42)
scaler_for_plot_tree = StandardScaler()
X_train_full_scaled_for_plot = scaler_for_plot_tree.fit_transform(X_train_full)
rf_model_for_plot.fit(X_train_full_scaled_for_plot, y_train_full)
individual_tree = rf_model_for_plot.estimators_[0]
'''
plt.figure(figsize=(20, 15))
plot_tree(individual_tree, feature_names=feature_names, class_names=class_names, filled=True, rounded=True, fontsize=10)
plt.title("Individual Decision Tree from Random Forest (fitted on full training data)")
plt.show()'''

# XGBoost
xgb_model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False,
                              random_state=42)
all_model_metrics_cv['XGBoost'], model_fold_accuracies['XGBoost'] = run_model_with_kfold(xgb_model, 'XGBoost', X_train_full, y_train_full, n_splits=10)

# Naive Bayes
nb_model = GaussianNB()
all_model_metrics_cv['Naive Bayes'], model_fold_accuracies['Naive Bayes'] = run_model_with_kfold(nb_model, 'Naive Bayes', X_train_full, y_train_full,
                                                           n_splits=10)

# Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
all_model_metrics_cv['Decision Tree'], model_fold_accuracies['Decision Tree'] = run_model_with_kfold(dt_model, 'Decision Tree', X_train_full, y_train_full,
                                                             n_splits=10)

# --- Final Model Evaluation on the Held-Out Test Set ---
print("\n\n--- Final Model Evaluation on Held-Out Test Set (using full training data for fitting) ---")
final_model_metrics = {}

scaler_final = StandardScaler()
X_train_full_scaled_final = scaler_final.fit_transform(X_train_full)
X_test_final_scaled_final = scaler_final.transform(X_test_final)

# KNN Final
knn_model_final = KNeighborsClassifier(n_neighbors=3)
knn_model_final.fit(X_train_full_scaled_final, y_train_full)
y_pred_knn_final = knn_model_final.predict(X_test_final_scaled_final)
final_model_metrics['KNN'] = _print_evaluation_metrics('KNN', y_test_final, y_pred_knn_final, is_final_test=True)

# Multi-Linear Regression Final
lr_model_final = LinearRegression()
lr_model_final.fit(X_train_full_scaled_final, y_train_full)
y_pred_continuous_final = lr_model_final.predict(X_test_final_scaled_final)
y_pred_lr_final = (y_pred_continuous_final > 0.5).astype(int)
final_model_metrics['Multi-Linear Regression'] = _print_evaluation_metrics('Multi-Linear Regression', y_test_final,
                                                                           y_pred_lr_final, is_final_test=True)

# Logistic Regression Final
logreg_model_final = LogisticRegression(max_iter=1000, random_state=42)
logreg_model_final.fit(X_train_full_scaled_final, y_train_full)
y_pred_logreg_final = logreg_model_final.predict(X_test_final_scaled_final)
final_model_metrics['Logistic Regression'] = _print_evaluation_metrics('Logistic Regression', y_test_final,
                                                                       y_pred_logreg_final, is_final_test=True)

# SVM
svm_model_final = SVC(kernel='rbf', random_state=42)
svm_model_final.fit(X_train_full_scaled_final, y_train_full)
y_pred_svm_final = svm_model_final.predict(X_test_final_scaled_final)
final_model_metrics['SVM'] = _print_evaluation_metrics('SVM', y_test_final, y_pred_svm_final, is_final_test=True)

# Random Forest Final
rf_model_final = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model_final.fit(X_train_full_scaled_final, y_train_full)
y_pred_rf_final = rf_model_final.predict(X_test_final_scaled_final)
final_model_metrics['Random Forest'] = _print_evaluation_metrics('Random Forest', y_test_final, y_pred_rf_final,
                                                                 is_final_test=True)

# XGBoost Final
xgb_model_final = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False,
                                    random_state=42)
xgb_model_final.fit(X_train_full_scaled_final, y_train_full)
y_pred_xgb_final = xgb_model_final.predict(X_test_final_scaled_final)
final_model_metrics['XGBoost'] = _print_evaluation_metrics('XGBoost', y_test_final, y_pred_xgb_final,
                                                           is_final_test=True)

# Naive Bayes Final
nb_model_final = GaussianNB()
nb_model_final.fit(X_train_full_scaled_final, y_train_full)
y_pred_nb_final = nb_model_final.predict(X_test_final_scaled_final)
final_model_metrics['Naive Bayes'] = _print_evaluation_metrics('Naive Bayes', y_test_final, y_pred_nb_final,
                                                               is_final_test=True)

# Decision Tree Final
dt_model_final = DecisionTreeClassifier(random_state=42)
dt_model_final.fit(X_train_full_scaled_final, y_train_full)
y_pred_dt_final = dt_model_final.predict(X_test_final_scaled_final)
final_model_metrics['Decision Tree'] = _print_evaluation_metrics('Decision Tree', y_test_final, y_pred_dt_final,
                                                                 is_final_test=True)

print("\n\n--- Comprehensive Summary of All Model Metrics (Cross-Validation Averages) ---")
for model_name, metrics in all_model_metrics_cv.items():
    print(f"\nModel: {model_name}")
    print(f"  Average Accuracy: {metrics['Average Accuracy']:.4f}")
    print(f"  Average Loss : {metrics['Average Loss']:.4f}")
    print(f"  Average Precision: {metrics['Average Precision']:.4f}")
    print(f"  Average Recall: {metrics['Average Recall']:.4f}")
    print(f"  Average F1-Score: {metrics['Average F1-Score']:.4f}")

print("\n\n--- Comprehensive Summary of All Model Metrics (Final Test Set Evaluation) ---")
for model_name, metrics in final_model_metrics.items():
    print(f"\nModel: {model_name}")
    print(f"  Accuracy: {metrics['Accuracy']:.4f}")
    print(f"  Loss : {metrics['Loss']:.4f}")
    print(f"  Precision: {metrics['Precision']:.4f}")
    print(f"  Recall: {metrics['Recall']:.4f}")
    print(f"  F1-Score: {metrics['F1-Score']:.4f}")
    print("  Confusion Matrix:")
    for row in metrics['Confusion Matrix']:
        print(f"    {row}")

x_vals = list(range(1, n_splits_for_cv + 1))

acc_rf = model_fold_accuracies['Random Forest']
acc_lr = model_fold_accuracies['Multi-Linear Regression']
acc_dt = model_fold_accuracies['Decision Tree']
acc_knn = model_fold_accuracies['KNN']
acc_nb = model_fold_accuracies['Naive Bayes']
acc_xgb = model_fold_accuracies['XGBoost']
acc_logreg = model_fold_accuracies['Logistic Regression']
acc_svm = model_fold_accuracies['SVM']

plot_data = [
    ("Random Forest", acc_rf, 'blue'),
    ("Multi-Linear Regression", acc_lr, 'green'),
    ("Decision Tree", acc_dt, 'purple'),
    ("K-Nearest Neighbors", acc_knn, 'orange'),
    ("Naive Bayes", acc_nb, 'red'),
    ("XGBoost", acc_xgb, 'cyan'),
    ("Logistic Regression", acc_logreg, 'brown'),
    ("SVM", acc_svm, 'magenta')
]

y_tick_values = [i * 0.05 for i in range(10, 21)]

fig, axes = plt.subplots(2, 4, figsize=(11, 5))
axes = axes.flatten()

for i, (model_name, accuracies, color) in enumerate(plot_data):
    ax = axes[i]
    ax.plot(x_vals, accuracies, marker='o', color=color, linestyle='--')
    ax.set_title(model_name)
    ax.set_xlabel("Fold Number")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.5, 1.05)
    ax.set_yticks(y_tick_values)
    ax.grid(True)

plt.tight_layout()
plt.suptitle("Model Accuracy Across K-Fold Cross-Validation Splits", y=1.02, fontsize=12)
plt.show()
