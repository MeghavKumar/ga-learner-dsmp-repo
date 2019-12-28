# --------------
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

#path : File path

# Code starts here


# read the dataset
dataset = pd.read_csv(path)


# look at the first five columns
dataset.head(5)


# Check if there's any column which is not useful and remove it like the column id
dataset.drop(['Id'],axis=1,inplace =True )
dataset_scaled = preprocessing.scale(dataset)
#X =  dataset.loc[: ,dataset.column!='Cover_Type']
#y = dataset['Cover_Type'].copy()
#X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)
# check the statistical description
dataset_scaled



# --------------
# We will visualize all the attributes using Violin Plot - a combination of box and density plots
import seaborn as sns
from matplotlib import pyplot as plt

#names of all the attributes 
cols = list(dataset.columns)
#print(cols)

#number of attributes (exclude target)
cols = dataset.columns
size = len(cols)-1
print(size)
#x-axis has target attribute to distinguish between classes
x = dataset['Cover_Type'].copy()


#y-axis shows values of an attribute
y = dataset.loc[:,dataset.columns!='Cover_Type']

#Plot violin for all attributes
for i in range(size):
    
    sns.violinplot(y = dataset.iloc[:,i]) 




# --------------
import numpy as np
upper_threshold = 0.5
lower_threshold = -0.5


# Code Starts Here
#Creating Subset of continous features

subset_train = dataset.iloc[: ,:10]
#print(subset_train.shape)

#Pearson Correlation
data_corr = subset_train.corr(method = 'pearson')

#projecting heatmap            
sns.heatmap(data_corr,xticklabels=data_corr.columns,yticklabels=data_corr.columns,annot=True,
linewidth=0.5)

#Finding Correlation
correlation = data_corr.unstack().sort_values(kind='quicksort')
#upper = data_corr.where(np.triu(np.ones(data_corr.shape), k=1).astype(np.bool))
#print('Upper',upper)
#lower = data_corr.where(np.tril(np.ones(data_corr.shape), k=1).astype(np.bool))
#print(lower)
#corr_var_list = [upper,lower]
#s = pd.Series(correlation)

#corr_var_list =correlation[((correlation>=upper_threshold)|(correlation<=lower_threshold) &(correlation != 1))]
#print(corr_var_list)
corr_var_list = correlation[(correlation>=upper_threshold)|(correlation<=lower_threshold)]
corr_var_list = corr_var_list[(correlation!=1)]
corr_var_list








 

# Code ends here




# --------------
#Import libraries 
from sklearn import cross_validation
from sklearn.preprocessing import StandardScaler

# Identify the unnecessary columns and remove it
dataset.drop(columns=['Soil_Type7', 'Soil_Type15'], inplace=True)

r,c = dataset.shape
X = dataset.iloc[:,:-1]
Y = dataset.iloc[:,-1]
# Scales are not the same for all variables. Hence, rescaling and standardization may be necessary for some algorithm to be applied on it.
X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size=0.2, random_state=0)
#Standardized
scaler = StandardScaler()
#Apply transform only for continuous data
X_train_temp = scaler.fit_transform(X_train.iloc[:,:10])
X_test_temp = scaler.transform(X_test.iloc[:,:10])
#Concatenate scaled continuous data and categorical
X_train1 = numpy.concatenate((X_train_temp,X_train.iloc[:,10:c-1]),axis=1)
X_test1 = numpy.concatenate((X_test_temp,X_test.iloc[:,10:c-1]),axis=1)
scaled_features_train_df = pd.DataFrame(X_train1, index=X_train.index, columns=X_train.columns)
scaled_features_test_df = pd.DataFrame(X_test1, index=X_test.index, columns=X_test.columns)


# --------------
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_classif
import numpy as np
from sklearn import feature_selection
# Write your solution here:
skb = SelectPercentile(score_func=f_classif,percentile=90)

predictors=skb.fit_transform(X_train1, Y_train)
scores = list(skb.scores_)
Features = np.asarray(X_train.columns)

list_of_tuples = list(zip(Features, scores)) 
dataframe =  pd.DataFrame(list_of_tuples,columns = ['Features', 'scores'])
dataframe.sort_values(by=['scores'],ascending  = False,inplace = True)
#s=dataframe.quantile(.90)
#print('S',s)
#top_k_predictors=list(dataframe.scores < np.percentile(dataframe.scores,90))
top_k_predictors = list(dataframe['Features'][:predictors.shape[1]])
print(list(top_k_predictors))
#top_k_predictors = dataframe.scores.quantile(0.9)
#top_k_predictors=list(dataframe[dataframe.scores <dataframe.scores.quantile(.90)])
#print(top_k_predictors)


# --------------
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score
import numpy as np


le= LogisticRegression()
clf1 = OneVsRestClassifier(le)
clf=OneVsRestClassifier(le)
#Fit the model
model_fit_all_features = clf1.fit(X_train,Y_train)

#making predictions
predictions_all_features = clf1.predict(X_test)
score_all_features = accuracy_score(Y_test,predictions_all_features)
#calculating accuracy
print('Accuracy Score',score_all_features)

model_fit_all_features = clf.fit(scaled_features_train_df[top_k_predictors],Y_train)
predictions_top_features = clf.predict(scaled_features_test_df[top_k_predictors])







score_top_features = accuracy_score(Y_test,predictions_top_features)
print('predictions_top_features',score_top_features)


