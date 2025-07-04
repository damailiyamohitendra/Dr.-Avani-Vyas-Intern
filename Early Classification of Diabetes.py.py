import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler   ,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier


df=pd.read_csv("data.csv")

for i in df.columns:
    if df[i].dtype=='object':
        l=LabelEncoder()
        df[i]=l.fit_transform(df[i])

X=df.drop("class",axis=1)
Y=df['class']

#for standadization
scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)

#train and test
X_train,x_test,y_train,y_test=train_test_split ( X_scaled,Y,test_size=0.2,random_state=42)


#logisctic regression 
print("\n")
model=LogisticRegression()
model.fit(X_train,y_train)
y_predict=model.predict(x_test)
accuracy=accuracy_score(y_test,y_predict)
print("accuracy of logistic : ",accuracy*100)
loss_log=100-(accuracy*100)
print("loss for logistic is : ", loss_log)
print("\n")


# random forest 
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print("accuracy of random forest is : " , acc*100)
loss_r=100-(acc*100)
print("loss for random forest is : ", loss_r)
print("\n")

# naive bayes
model_naive=GaussianNB()
model_naive.fit(X_train,y_train)
pred_bayes=model_naive.predict(x_test)
accuracy_naive=accuracy_score(y_test,pred_bayes)
print("accuracy of naive bayes :", accuracy_naive*100)
loss_n=100-(accuracy_naive*100)
print("loss for naive bayes is : ", loss_n)
print("\n")

#decision tree
model_dec=DecisionTreeClassifier(random_state=42)
model_dec.fit(X_train,y_train)
pred_dec=model_dec.predict(x_test)
accuracy_dec=accuracy_score(y_test,pred_dec)
print("accuracy of decision tree :", accuracy_dec*100)
loss_d=100-(accuracy_dec*100)
print("loss for decision tree is : ", loss_d)
print("\n")



# grid search  with random forest
param_grid = {'n_estimators': [10,50,100]   ,  'max_depth': [5, 10],
            'min_samples_split': [2, 5,6,7] ,  'min_samples_leaf': [1, 2,3,4]        
}

r_with_g = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(
    r_with_g, param_grid,
    cv=5,          
    scoring='accuracy'      
)
grid_search.fit(X_train,y_train)

best_model = grid_search.best_estimator_
grid_pred = best_model.predict(x_test)
accuracy_grid = accuracy_score(y_test, grid_pred)
print("Accuracy with random forest with grid search : ", accuracy_grid * 100)
loss=100-(accuracy_grid*100)
print("loss for grid is : ", loss)
print("\n")