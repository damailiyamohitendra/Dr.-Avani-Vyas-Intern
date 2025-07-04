import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

try:
    df = pd.read_csv('D1 Pima Indians Diabetes (PIDD).csv')  # prints the whole dataset in one go
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

x = df.drop('Outcome', axis=1)
y = df['Outcome']

print(f'\nFeatures shape: {x.shape}')
print(f"Target shape: {y.shape}")

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

print(f"\nTraining features (X_train) shape: {x_train.shape}")
print(f"Testing features (X_test) shape: {x_test.shape}")
print(f"Training target (y_train) shape: {y_train.shape}")
print(f"Testing target (y_test) shape: {y_test.shape}")

model= LinearRegression()

model.fit(x_train, y_train)

print("\nMulti-Linear Regression model trained successfully!")
print(f"Model Coefficients: {model.coef_}")
print(f"Model Intercept: {model.intercept_:.4f}")

y_pred_continuous = model.predict(x_test)
print("\nContinuous predictions made on the test set...")

y_pred_binary = (y_pred_continuous > 0.5).astype(int)
print("Continuous predictions thresholded to binary (0 or 1) for classification metrics.")

accuracy = accuracy_score(y_test, y_pred_binary)
print(f"Accuracy: {accuracy:.4f}")
precision = precision_score(y_test, y_pred_binary)
print(f"Precision: {precision:.4f}")
recall = recall_score(y_test, y_pred_binary)
print(f"Recall: {recall:.4f}")
f1 = f1_score(y_test, y_pred_binary)
print(f"F1-Score: {f1:.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_binary))
