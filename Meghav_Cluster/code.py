# --------------
# import packages

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 



# Load Offers
offers = pd.read_excel(path,sheet_name=0)

# Load Transactions
transactions = pd.read_excel(path,sheet_name = 1)

transactions['n']=1

# Merge dataframes
df=pd.merge(offers, transactions)

# Look at the first 5 rows
df.head(5)


# --------------
# Code starts here

# create pivot table
matrix = pd.pivot_table(df,values='n',index='Customer Last Name',columns='Offer #',)
print(matrix.head())
# replace missing values with 0
matrix.fillna(0,inplace =True)

# reindex pivot table
matrix.reset_index(inplace=True)

# display first 5 rows
print(matrix.head(5))
# Code ends here


# --------------
# import packages
from sklearn.cluster import KMeans

# Code starts here

# initialize KMeans object
cluster= KMeans(n_clusters=5,init='k-means++',max_iter=300, n_init=10,random_state=0)

# create 'cluster' column
matrix['cluster']= cluster.fit_predict(matrix[matrix.columns[1:]])
print(matrix.head(5))

# Code ends here


# --------------
# import packages
from sklearn.decomposition import PCA

# Code starts here

# initialize pca object with 2 components
pca = PCA(n_components=2,random_state=0)

# create 'x' and 'y' columns donoting observation locations in decomposed form
matrix['x'] = pca.fit_transform(matrix[matrix.columns[1:]])[:,0]
matrix['y'] = pca.fit_transform(matrix[matrix.columns[1:]])[:,1]

# dataframe to visualize clusters by customer names
clusters = matrix.iloc[:,[0, 33, 34, 35]]

# visualize clusters

plt.scatter(x='x', y='y',c='cluster', cmap='viridis',data=clusters)
plt.show()

# Code ends here


# --------------
# Code starts here

# merge 'clusters' and 'transactions'
data = clusters.merge(transactions,on='Customer Last Name')

# merge `data` and `offers`
data = data.merge(offers)
print(data.head())

# initialzie empty dictionary
champagne = {}

# iterate over every cluster
for i in range(0,5):
    new_df=data[data['cluster']== i]

    # sort cluster according to type of 'Varietal'
    counts = new_df['Varietal'].value_counts(ascending=False)

    # check if 'Champagne' is ordered mostly
    if counts.index[0]=='Champagne':

        # add it to 'champagne'
        champagne.update({i: counts[0]})

# get cluster with maximum orders of 'Champagne' 
cluster_champagne = max(champagne, key=champagne.get)
print('Cluster_Champagne',cluster_champagne)







# --------------
# Code starts here

# empty dictionary
discount = {}

# iterate over cluster numbers
for j in range(0,5):

    # dataframe for every cluster
    new_df = data[data['cluster']== j]

    # average discount for cluster
    counts=new_df['Discount (%)'].mean()

    # adding cluster number as key and average discount as value 
    discount.update({j :counts})


# cluster with maximum average discount
cluster_discount = max(discount, key=discount.get)
print('Cluster_Discount',cluster_discount)

# Code ends here


