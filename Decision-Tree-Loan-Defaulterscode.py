# --------------
#Importing header files

import pandas as pd
from sklearn.model_selection import train_test_split


# Code starts here
data=pd.read_csv(path)

# Create Train Features
X = data.drop(['customer.id','paid.back.loan'],axis = 1)
# Create Target Features
y = data['paid.back.loan'].copy()

# split the data

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state = 0)
# Code ends here


# --------------
#Importing header files
import matplotlib.pyplot as plt

# Code starts here
#print('Y_train',y_train)
fully_paid = y_train.value_counts()
print('Fully Paid',fully_paid)

#plot bar graph
plt.bar(fully_paid,y.value_counts())
#plt.show()

# Code ends here


# --------------
#Importing header files
import numpy as np
from sklearn.preprocessing import LabelEncoder


# Code starts here
#Convert the column values to Float
X_train['int.rate']=X_train['int.rate'].replace('%','',regex=True).astype(float)
X_train['int.rate'] = X_train['int.rate']/100
#print(X_train['int.rate'])

X_test['int.rate']=X_test['int.rate'].replace('%','',regex=True).astype(float)
X_test['int.rate'] = X_test['int.rate']/100
#print(X_test['int.rate'])

#Create Numerical Subset of X_train
num_df=X_train.select_dtypes(include=np.number)


#Create Categorical  Subset of X_train
cat_df= X_train.select_dtypes(include=np.object)



# Code ends here


# --------------
#Importing header files
import seaborn as sns


# Code starts here

#Saving  Column names

cols = list(num_df.columns)
print('cols',cols)

fig ,axes = plt.subplots(nrows = 9 , ncols = 1)

for i in range(0,9):
    sns.boxplot(x=y_train, y=num_df[cols[i]],ax=axes[i])

# Code ends here


# --------------
# Code starts here

#Add column names

cols = list(cat_df.columns)

#Create Subplots
fig ,axes = plt.subplots(nrows = 2 , ncols = 2)

#Create Countplot
for i in range(0,2):
    for j in range(0,2):
        sns.countplot(x=X_train[cols[i*2+j]],hue=y_train,ax=axes[i,j])


# Code ends here


# --------------
#Importing header files
from sklearn.tree import DecisionTreeClassifier
for column in cat_df:
    X_train[column].fillna('NA')
    X_test[column].fillna('NA')


categorical_cols =list(cat_df.columns)
le = LabelEncoder()
X_train[categorical_cols] = X_train[categorical_cols].apply(lambda col: le.fit_transform(col))
X_test[categorical_cols] = X_test[categorical_cols].apply(lambda col: le.fit_transform(col))

#yt = y_train.to_frame()
y_train=y_train.replace({'No': 0, 'Yes': 1})
y_test=y_test.replace({'No': 0, 'Yes': 1})

model = DecisionTreeClassifier()
model.fit(X_train,y_train)

acc = model.score(X_test,y_test)
print('Accuracy',acc)
    



    



# Code ends here


# --------------
#Importing header files
from sklearn.model_selection import GridSearchCV

#Parameter grid
parameter_grid = {'max_depth': np.arange(3,10), 'min_samples_leaf': range(10,50,10)}

# Code starts here
# Object of Decision Tree
model_2 = DecisionTreeClassifier(random_state=0)
#Grid Search for pruning
p_tree = GridSearchCV(estimator=model_2,param_grid=parameter_grid,cv=5)
p_tree.fit(X_train,y_train)

acc_2 = p_tree.score(X_test,y_test)
print('New Accouracy',acc_2)


# Code ends here


# --------------
#Importing header files

from io import StringIO
from sklearn.tree import export_graphviz
from sklearn import tree
from sklearn import metrics
from IPython.display import Image
import pydotplus

# Code starts here
dot_data = export_graphviz(decision_tree=p_tree.best_estimator_, out_file=None, feature_names=X.columns, filled = True ,class_names=['loan_paid_back_yes','loan_paid_back_no'])

graph_big = pydotplus.graph_from_dot_data(dot_data)

# show graph - do not delete/modify the code below this line
img_path = user_data_dir+'/file.png'
graph_big.write_png(img_path)

plt.figure(figsize=(20,15))
plt.imshow(plt.imread(img_path))
plt.axis('off')
plt.show() 

# Code ends here


