import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
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
    loss = 1-accuracy
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
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
        'Loss' : loss,
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

print("\n --- Training Multi-Linear Regression ---")
model= LinearRegression()
model.fit(X_train, y_train)
print(f"Model Coefficients: {model.coef_}")
print(f"Model Intercept: {model.intercept_:.4f}")

y_pred_continuous = model.predict(X_test)
y_pred_binary = (y_pred_continuous > 0.5).astype(int)
all_model_metrics['Multi-Linear Regression'] = print_metrics('Multi-Linear Regression', y_test, y_pred_binary)

print("\n--- Training Random Forest ---")
rf_model = RandomForestClassifier(random_state=42)
'''param_grid = {
    'n_estimators': [10, 50, 100],
    'max_depth': [5, 10],
    'min_samples_split': [2, 5, 6, 7],
    'min_samples_leaf': [1, 2, 3, 4]
}

print("Parameter grid to search:")
for param, values in param_grid.items():
    print(f"  {param}: {values}")

print("\n5. Initializing GridSearchCV with 'accuracy' as the scoring metric...")
grid_search = GridSearchCV(
    estimator=rf_model,
    param_grid=param_grid,
    scoring='accuracy',
    cv=5,
    verbose=2,
)
print("\n6. Starting Grid Search (this may take a moment, evaluating multiple combinations)...")
grid_search.fit(X_train, y_train)

print("\n7. Grid Search completed.")
print(f"Best parameters found: {grid_search.best_params_}")
print(f"Best cross-validation accuracy (maximized by Grid Search): {grid_search.best_score_:.4f}")

best_rf_model = grid_search.best_estimator_

print("\n8. Evaluating the best Random Forest model on the test data...")
y_pred_rf_best = best_rf_model.predict(X_test)'''

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
    print(f"  Loss : {metrics['Loss']:.4f}")
    print(f"  Precision: {metrics['Precision']:.4f}")
    print(f"  Recall: {metrics['Recall']:.4f}")
    print(f"  F1-Score: {metrics['F1-Score']:.4f}")
    print("  Confusion Matrix:")
    for row in metrics['Confusion Matrix']:
        print(f"    {row}")