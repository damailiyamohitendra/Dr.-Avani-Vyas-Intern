import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
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

x = df.drop(columns=['Outcome','Age', 'Skin thickness', 'Body mass index', 'Insulin'], axis=1)
y = df['Outcome']

print(f'\nFeatures shape: {x.shape}')
print(f"Target shape: {y.shape}")

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42)

print(f"\nTraining features (X_train) shape: {x_train.shape}")
print(f"Testing features (X_test) shape: {x_test.shape}")
print(f"Training target (y_train) shape: {y_train.shape}")
print(f"Testing target (y_test) shape: {y_test.shape}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(x_train)
X_test_scaled = scaler.transform(x_test)

print("\nFeatures scaled successfully.")

k_values = range(1, 25)
print("\nEvaluating KNN for different 'k' values")

results = []

for iter, k in enumerate(k_values):
    print(f"\nIteration {iter + 1}: Evaluating with k = {k} ")

    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)

    y_pred = knn.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print("  Confusion Matrix: ")
    print(confusion_matrix(y_test, y_pred))

    results.append({
        'k': k,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    })

print("\nSummary of Results for Different 'k' Values")
for res in results:
    print(f"k={res['k']:<2}: Acc={res['accuracy']:.4f}, Prec={res['precision']:.4f}, Rec={res['recall']:.4f}, F1={res['f1_score']:.4f}")

# Find the best k based on accuracy
best_k_accuracy = max(results, key=lambda x: x['accuracy'])
print(f"\nBest 'k' based on Accuracy: {best_k_accuracy['k']} (Accuracy: {best_k_accuracy['accuracy']:.4f})")

'''metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
metrics_values = [accuracy, precision, recall, f1]

plt.figure(figsize=(8, 6))
bars = plt.bar(metrics_names, metrics_values, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
plt.ylim(0, 1)
plt.title('KNN Model Performance Metrics', fontsize=14)
plt.ylabel('Score', fontsize=12)
plt.xlabel('Metric', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, round(yval, 4), ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()
'''
