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

df = pd.read_csv('D1 Pima Indians Diabetes (PIDD).csv')
for col in df.select_dtypes(include='object').columns:
    df[col] = LabelEncoder().fit_transform(df[col])

x = df.drop('Outcome', axis=1)
y = df['Outcome']


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
