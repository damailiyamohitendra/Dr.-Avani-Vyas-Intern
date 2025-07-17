import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import shap
import matplotlib.pyplot as plt
import seaborn as sns

# Load and preprocess data
df1 = pd.read_csv('combined_dataset.csv')
df2 = pd.read_csv('Pima_Indians_Diabetes_PIDD_train_balancedSMOTE.csv')
df3 = pd.read_csv('female_only Diabetes Prediction.csv')
df4 = pd.read_csv('DataSet_P_Drop.csv')
df4 = df4[df4['Gender'] != 1].copy()

for df in [df1, df2, df3, df4]:
    df.columns = df.columns.str.strip()
    df.reset_index(drop=True, inplace=True)

class_col = [col for col in df4.columns if 'CLASS' in col.upper()]
if not class_col:
    raise ValueError("No column found that looks like 'CLASS' in df4")

diabetes_2 = df2['Outcome']
diabetes_3 = df3['diabetes']
diabetes_4 = df4[class_col[0]]
diabetes_all = pd.concat([diabetes_2, diabetes_3, diabetes_4], ignore_index=True)
df_combined = pd.DataFrame({'diabetes': diabetes_all})
df_combined.to_csv('COMBINED_DATASET_K.csv', index=False)

features_2 = df2.drop(columns=['Outcome'])
features_3 = df3.drop(columns=['diabetes'])
features_4 = df4.drop(columns=[class_col[0]])
X = pd.concat([features_2, features_3, features_4], ignore_index=True)
y = df_combined['diabetes'].astype(int)

for col in X.columns:
    if X[col].dtype == 'object':
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

X = pd.DataFrame(SimpleImputer(strategy='mean').fit_transform(X), columns=X.columns)
X_scaled = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, stratify=y, test_size=0.2, random_state=42)

models = {
    "Logistic Regression": LogisticRegression(max_iter=5000),
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(probability=True),
    "Naive Bayes": GaussianNB(),
    "Multi-Linear Regression": LinearRegression(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    if name == "Multi-Linear Regression":
        y_pred_raw = model.predict(X_test)
        y_pred = pd.Series([int(round(pred)) for pred in y_pred_raw]).clip(lower=y.min(), upper=y.max())
    else:
        y_pred = model.predict(X_test)
    y_pred = pd.Series(y_pred).astype(int)
    acc = accuracy_score(y_test, y_pred)
    loss = 1 - acc
    prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    return {
        "Model": name,
        "Accuracy": round(acc, 4),
        "Loss": round(loss, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1 Score": round(f1, 4)
    }

results = [evaluate_model(name, model, X_train, X_test, y_train, y_test) for name, model in models.items()]
results_df = pd.DataFrame(results)
print(results_df)

metrics = ["Accuracy", "Loss", "Precision", "Recall", "F1 Score"]
for metric in metrics:
    plt.figure(figsize=(10, 4))
    sns.barplot(x='Model', y=metric, data=results_df)
    plt.title(f"{metric} Comparison Across Models")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

xgb_model = models["XGBoost"]
xgb_model.fit(X_train, y_train)
explainer = shap.Explainer(xgb_model)
shap_values = explainer(X_test)
shap.summary_plot(shap_values, pd.DataFrame(X_test, columns=X.columns))
