import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

try:
    df = pd.read_csv('sylhet diabetes data.csv')  # prints the whole dataset in one go
    print("Dataset loaded!")
    print(df)
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
print(f"Dataset shape: X={X.shape}, y={y.shape} (Classification)")

print("\n2. Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training data shape: X_train={X_train.shape}, y_train={y_train.shape}")
print(f"Testing data shape: X_test={X_test.shape}, y_test={y_test.shape}")

print("\n3. Applying feature scaling...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Features scaled successfully.")

all_model_metrics = {}

def print_metrics(model_name, y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n--- {model_name} Metrics ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)

    return {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Confusion Matrix': cm.tolist()
    }


print("\n--- Training K-Nearest Neighbors (KNN) ---")
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)
y_pred_knn = knn_model.predict(X_test_scaled)
all_model_metrics['KNN'] = print_metrics('KNN', y_test, y_pred_knn)
plt.show()

print("\n--- Training Random Forest ---")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
all_model_metrics['Random Forest'] = print_metrics('Random Forest', y_test, y_pred_rf)
plt.show()

print("\n--- Training XGBoost ---")
xgb_model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False, random_state=42)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
all_model_metrics['XGBoost'] = print_metrics('XGBoost', y_test, y_pred_xgb)
plt.show()

print("\n--- Training Naive Bayes ---")
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)
all_model_metrics['Naive Bayes'] = print_metrics('Naive Bayes', y_test, y_pred_nb)
plt.show()

print("\n--- Training Decision Tree ---")
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)
all_model_metrics['Decision Tree'] = print_metrics('Decision Tree', y_test, y_pred_dt)
plt.show()

print("\n\n--- Comprehensive Summary of All Model Metrics ---")
for model_name, metrics in all_model_metrics.items():
    print(f"\nModel: {model_name}")
    print(f"  Accuracy: {metrics['Accuracy']:.4f}")
    print(f"  Precision: {metrics['Precision']:.4f}")
    print(f"  Recall: {metrics['Recall']:.4f}")
    print(f"  F1-Score: {metrics['F1-Score']:.4f}")
    print("  Confusion Matrix:")
    for row in metrics['Confusion Matrix']:
        print(f"    {row}")