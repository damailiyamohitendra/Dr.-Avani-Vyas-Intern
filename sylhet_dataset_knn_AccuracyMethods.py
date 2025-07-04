import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import xgboost as xgb
from tensorflow import keras
from tensorflow.keras import layers

try:
    df = pd.read_csv('sylhet diabetes data.csv')
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

print(f"Dataset shape: X={X.shape}, y={y.shape}")

print("\n2. Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
print(f"Training data shape: X_train={X_train.shape}, y_train={y_train.shape}")
print(f"Testing data shape: X_test={X_test.shape}, y_test={y_test.shape}")

print("\n3. Applying feature scaling...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Features scaled successfully.")

results = {}

print("\n--- Training K-Nearest Neighbors (KNN) ---")
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)
y_pred_knn = knn_model.predict(X_test_scaled)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
results['KNN'] = accuracy_knn
print(f"KNN Accuracy: {accuracy_knn:.4f}")

print("\n--- Training Random Forest ---")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
results['Random Forest'] = accuracy_rf
print(f"Random Forest Accuracy: {accuracy_rf:.4f}")

print("\n--- Training XGBoost ---")
xgb_model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False, random_state=42)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
results['XGBoost'] = accuracy_xgb
print(f"XGBoost Accuracy: {accuracy_xgb:.4f}")

print("\n--- Training Naive Bayes ---")
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)
accuracy_nb = accuracy_score(y_test, y_pred_nb)
results['Naive Bayes'] = accuracy_nb
print(f"Naive Bayes Accuracy: {accuracy_nb:.4f}")

print("\n--- Training Decision Tree ---")
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)
accuracy_dt = accuracy_score(y_test, y_pred_dt)
results['Decision Tree'] = accuracy_dt
print(f"Decision Tree Accuracy: {accuracy_dt:.4f}")

print("\n--- Training Artificial Neural Network (ANN) ---")
ann_model = keras.Sequential([
    layers.Input(shape=(X_train_scaled.shape[1],)), # Input layer matching number of features
    layers.Dense(64, activation='relu'),            # Hidden layer with ReLU activation
    layers.Dropout(0.3),                            # Dropout for regularization
    layers.Dense(32, activation='relu'),            # Another hidden layer
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')           # Output layer for binary classification
])

ann_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history = ann_model.fit(
    X_train_scaled, y_train,
    epochs=50,              # Number of training epochs
    batch_size=32,          # Number of samples per gradient update
    verbose=0,              # Suppress verbose output during training
    validation_split=0.2    # Use 20% of training data for validation
)

loss_ann, accuracy_ann = ann_model.evaluate(X_test_scaled, y_test, verbose=0)
results['ANN'] = accuracy_ann
print(f"ANN Accuracy: {accuracy_ann:.4f}")

print("\n--- Summary of Model Accuracies ---")
for model_name, accuracy in results.items():
    print(f"{model_name}: {accuracy:.4f}")