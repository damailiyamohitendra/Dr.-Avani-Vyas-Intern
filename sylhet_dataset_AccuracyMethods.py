'''import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier # Import for Decision Tree
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

try:
    df = pd.read_csv('sylhet diabetes data.csv')
    print("Dataset loaded!")
    # Changed from printing the whole df to .head() for brevity in output
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

X = df.drop(columns=['class'], axis=1)
y = df['class']
print(f"\nDataset shape: X={X.shape}, y={y.shape} (Classification)")

all_model_metrics = {}

def print_metrics(model_name, y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    loss = 1 - accuracy
    precision = precision_score(y_true, y_pred, zero_division=0) # Added zero_division to avoid warning
    recall = recall_score(y_true, y_pred, zero_division=0) # Added zero_division to avoid warning
    f1 = f1_score(y_true, y_pred, zero_division=0) # Added zero_division to avoid warning
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n--- {model_name} Metrics ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Loss (Misclassification Rate): {loss:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)

    return {
        'Accuracy': accuracy,
        'Loss': loss,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Confusion Matrix': cm.tolist()
    }

model_accuracies_for_plot = {
    'KNN': [],
    'Multi-Linear Regression': [],
    'Random Forest': [],
    'XGBoost': [],
    'Naive Bayes': [],
    'Decision Tree': []
}

X_train_iter, X_test_iter, y_train_iter, y_test_iter = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler_iter = StandardScaler()
X_train_iter_scaled = scaler_iter.fit_transform(X_train_iter)
X_test_iter_scaled = scaler_iter.transform(X_test_iter)

# --- KNN ---
knn_model_iter = KNeighborsClassifier()
knn_model_iter.fit(X_train_iter_scaled, y_train_iter)
y_pred_knn_iter = knn_model_iter.predict(X_test_iter_scaled)
model_accuracies_for_plot['KNN'].append(accuracy_score(y_test_iter, y_pred_knn_iter))

# --- Multi-Linear Regression ---
lr_model_iter = LinearRegression()
lr_model_iter.fit(X_train_iter_scaled, y_train_iter)
y_pred_continuous_lr_iter = lr_model_iter.predict(X_test_iter_scaled)
y_pred_lr_iter = (y_pred_continuous_lr_iter > 0.5).astype(int)
model_accuracies_for_plot['Multi-Linear Regression'].append(accuracy_score(y_test_iter, y_pred_lr_iter))

# --- Random Forest ---
rf_model_iter = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model_iter.fit(X_train_iter_scaled, y_train_iter)
y_pred_rf_iter = rf_model_iter.predict(X_test_iter_scaled)
model_accuracies_for_plot['Random Forest'].append(accuracy_score(y_test_iter, y_pred_rf_iter))

# --- XGBoost ---
xgb_model_iter = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False, random_state=42)
xgb_model_iter.fit(X_train_iter_scaled, y_train_iter)
y_pred_xgb_iter = xgb_model_iter.predict(X_test_iter_scaled)
model_accuracies_for_plot['XGBoost'].append(accuracy_score(y_test_iter, y_pred_xgb_iter))

# --- Naive Bayes ---
nb_model_iter = GaussianNB()
nb_model_iter.fit(X_train_iter_scaled, y_train_iter)
y_pred_nb_iter = nb_model_iter.predict(X_test_iter_scaled)
model_accuracies_for_plot['Naive Bayes'].append(accuracy_score(y_test_iter, y_pred_nb_iter))

# --- Decision Tree ---
dt_model_iter = DecisionTreeClassifier(random_state=42)
dt_model_iter.fit(X_train_iter_scaled, y_train_iter)
y_pred_dt_iter = dt_model_iter.predict(X_test_iter_scaled)
model_accuracies_for_plot['Decision Tree'].append(accuracy_score(y_test_iter, y_pred_dt_iter))

x_vals = list(range(11))

plot_data = [
    ("Random Forest", model_accuracies_for_plot['Random Forest'], 'blue'),
    ("Multi-Linear Regression", model_accuracies_for_plot['Multi-Linear Regression'], 'green'),
    ("Decision Tree", model_accuracies_for_plot['Decision Tree'], 'purple'),
    ("K-Nearest Neighbors", model_accuracies_for_plot['KNN'], 'orange'),
    ("Naive Bayes", model_accuracies_for_plot['Naive Bayes'], 'red'),
    ("XGBoost", model_accuracies_for_plot['XGBoost'], 'cyan')
]
y_tick_values = np.arange(0.5, 1.05, 0.05)
fig, axes = plt.subplots(2, 3, figsize=(10, 8))
axes = axes.flatten()

for i, (model_name, accuracies, color) in enumerate(plot_data):
    ax = axes[i]
    ax.plot(x_vals, accuracies, marker='o', color=color, linestyle='--')
    ax.set_title(model_name)
    ax.set_xlabel("Iteration Number")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.5, 1.05)
    ax.set_yticks(y_tick_values)
    ax.grid(True)

plt.tight_layout()
plt.suptitle("Model Accuracy Across 10 Independent Train-Test Splits", y=1.00, fontsize=14)
plt.show()'''

import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression

df = pd.read_csv('sylhet diabetes data.csv')
for col in df.select_dtypes(include='object').columns:
    df[col] = LabelEncoder().fit_transform(df[col])

x = df.drop('class', axis=1)
y = df['class']


scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)


x_vals = list(range(11))
acc_rf = []
acc_xg = []
acc_log = []
acc_nb = []
acc_decision = []
acc_knn = []
acc_svm=[]
acc_multi=[]


for i in x_vals:
    model_rf = RandomForestClassifier(random_state=i)
    model_rf.fit(x_train, y_train)
    acc_rf.append(accuracy_score(y_test, model_rf.predict(x_test)))

    model_xg = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False,
                                       random_state=42)
    model_xg.fit(x_train, y_train)
    acc_xg.append(accuracy_score(y_test, model_xg.predict(x_test)))

    model_log = LogisticRegression(max_iter=1000, random_state=i)
    model_log.fit(x_train, y_train)
    acc_log.append(accuracy_score(y_test, model_log.predict(x_test)))

    model_nb = GaussianNB()
    model_nb.fit(x_train, y_train)
    acc_nb.append(accuracy_score(y_test, model_log.predict(x_test)))

    model_dec = DecisionTreeClassifier(random_state=i)
    model_dec.fit(x_train, y_train)
    acc_decision.append(accuracy_score(y_test, model_dec.predict(x_test)))

    model_knn = KNeighborsClassifier()
    model_knn.fit(x_train, y_train)
    acc_knn.append(accuracy_score(y_test, model_knn.predict(x_test)))

    model_svm= SVC()
    model_svm.fit(x_train,y_train)
    acc_svm.append(accuracy_score(y_test,model_svm.predict(x_test)))

    model_multi = LinearRegression()
    model_multi.fit(x_train, y_train)
    y_pred_multi = model_multi.predict(x_test)
    y_pred_class = (y_pred_multi >= 0.5).astype(int)
    acc_multi.append(accuracy_score(y_test, y_pred_class))


fig, axis = plt.subplots(2, 4, figsize=(10, 8))


axis[0, 0].plot(x_vals, acc_rf, marker='o', color='blue', linestyle='--')
axis[0, 0].set_title("Random Forest")
axis[0, 0].set_ylim(0.5, 1.05)
axis[0, 0].set_yticks(np.arange(0.5, 1.05, 0.05))
axis[0, 0].grid(True)

axis[0, 1].plot(x_vals, acc_xg, marker='o', color='green', linestyle='--')
axis[0, 1].set_title("XGBoost")
axis[0, 1].set_ylim(0.5, 1.05)
axis[0, 1].set_yticks(np.arange(0.5, 1.05, 0.05))
axis[0, 1].grid(True)

axis[0, 2].plot(x_vals, acc_xg, marker='o', color='green', linestyle='--')
axis[0, 2].set_title("Logistic Regression")
axis[0, 2].set_ylim(0.5, 1.05)
axis[0, 2].set_yticks(np.arange(0.5, 1.05, 0.05))
axis[0, 2].grid(True)

axis[0, 3].plot(x_vals, acc_xg, marker='o', color='green', linestyle='--')
axis[0, 3].set_title("Naive Bayes")
axis[0, 3].set_ylim(0.5, 1.05)
axis[0, 3].set_yticks(np.arange(0.5, 1.05, 0.05))
axis[0, 3].grid(True)

axis[1, 0].plot(x_vals, acc_svm, marker='o', color='green', linestyle='--')
axis[1, 0].set_title("SVM")
axis[1, 0].set_ylim(0.5, 1.05)
axis[1, 0].set_yticks(np.arange(0.5, 1.05, 0.05))
axis[1, 0].grid(True)

axis[1, 1].plot(x_vals, acc_decision, marker='o', color='purple', linestyle='--')
axis[1, 1].set_title("Decision Tree")
axis[1, 1].set_ylim(0.5, 1.05)
axis[1, 1].set_yticks(np.arange(0.5, 1.05, 0.05))
axis[1, 1].grid(True)

axis[1, 2].plot(x_vals, acc_multi, marker='o', color='orange', linestyle='--')
axis[1, 2].set_title("Multi-Linear")
axis[1, 2].set_ylim(0.5, 1.05)
axis[1, 2].set_yticks(np.arange(0.5, 1.05, 0.05))
axis[1, 2].grid(True)

axis[1, 3].plot(x_vals, acc_knn, marker='o', color='orange', linestyle='--')
axis[1, 3].set_title("K-Nearest Neighbors")
axis[1, 3].set_ylim(0.5, 1.05)
axis[1, 3].set_yticks(np.arange(0.5, 1.05, 0.05))
axis[1, 3].grid(True)


plt.suptitle("Model Accuracy plot", fontsize=16, y=1)
plt.tight_layout()
plt.show()
