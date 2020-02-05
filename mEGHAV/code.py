# --------------
import pandas as pd
from sklearn.model_selection import train_test_split
#path - Path of file 
df=pd.read_csv(path)
df.head()

# Code starts here
#Adding Features
X = df.drop(['customerID','Churn'],1)

#Adding Target
y = df['Churn']

#Splitting the data
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)





# --------------
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Code starts here
# Replace spaces
#X_train['TotalCharges']=X_train['TotalCharges'].replace(to_replace = '', value = np.NaN)
#X_train['TotalCharges']=(X_train['TotalCharges'].str.split()).apply(lambda x: float(x[0].replace(' ',np.NaN)))
X_train['TotalCharges'] = X_train['TotalCharges'].apply(lambda x: np.nan if isinstance(x, str) and x.isspace() else x)
X_test['TotalCharges'] = X_test['TotalCharges'].apply(lambda x: np.nan if isinstance(x, str) and x.isspace() else x)


#Change datatype 
X_train['TotalCharges']=X_train['TotalCharges'].astype(float)
X_test['TotalCharges']=X_test['TotalCharges'].astype(float)

#Fill the missing values
X_train['TotalCharges']=X_train['TotalCharges'].fillna(X_train['TotalCharges'].mean())
X_test['TotalCharges']=X_test['TotalCharges'].fillna(X_test['TotalCharges'].mean())

#Checking null values
X_train['TotalCharges'].isnull().sum()

#identify categorical features
categorical_cols = X_train.select_dtypes(exclude=np.number).columns.tolist()

# instantiate labelencoder object
le = LabelEncoder()

# apply le on categorical feature columns
X_train[categorical_cols] = X_train[categorical_cols].apply(lambda col: le.fit_transform(col))
X_test[categorical_cols] = X_test[categorical_cols].apply(lambda col: le.fit_transform(col))

# Replace target values
y_train.replace(('Yes', 'No'), (1, 0), inplace=True)
print(y_train)
y_test.replace(('Yes', 'No'), (1, 0), inplace=True)




# --------------
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

# Code starts here
print('X_train',X_train)
print('X_test',X_test)
print('y_train',y_train)
print('y_test',y_test)

#Applying Adaboost on Weak Learner
ada_model = AdaBoostClassifier(random_state=0)

#Train the model
ada_model.fit(X_train,y_train)

#Making predictions
y_pred = ada_model.predict(X_test)
print('Predictions',y_pred)

#Calculate Accuracy
ada_score = accuracy_score(y_test,y_pred)
print('Accuracy',ada_score)

#Calculate Confusion Matrix
ada_cm = confusion_matrix(y_test,y_pred)
print('Confusion Matrix',ada_cm)

#Make Classification Report
ada_cr = classification_report(y_test,y_pred)
print('Classification Report',ada_cr)



# --------------
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

#Parameter list
parameters={'learning_rate':[0.1,0.15,0.2,0.25,0.3],
            'max_depth':range(1,3)}

# Code starts here
#Applying XGboost on Weak Learner
xgb_model = XGBClassifier(random_state=0)

#Train the model
xgb_model.fit(X_train,y_train)

#Make Prediction
y_pred= xgb_model.predict(X_test)
print('Prediction Score',y_pred)

#Accuracy Score
xgb_score = accuracy_score(y_test,y_pred)
print('Accuracy Score',xgb_score)

#Make Confusion Matrix
xgb_cm = confusion_matrix(y_test,y_pred)
print('Confusion Matrix', xgb_cm)

#Classification Report
xgb_cr = classification_report(y_test,y_pred)
print('Cliassification Report:',xgb_cr)

#Applying Grid Search
clf_model = GridSearchCV(estimator=xgb_model,param_grid=parameters)

#Train the model
clf_model.fit(X_train,y_train)

#Make Prediction
y_pred= clf_model.predict(X_test)
print('Prediction Score',y_pred)

#Accuracy Score
clf_score = accuracy_score(y_test,y_pred)
print('Accuracy Score',clf_score)

#Make Confusion Matrix
clf_cm = confusion_matrix(y_test,y_pred)
print('Confusion Matrix', clf_cm)

#Classification Report
clf_cr = classification_report(y_test,y_pred)
print('Cliassification Report:',clf_cr)




