# --------------
#Importing header files
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Code starts here
data = pd.read_csv(path)

#plot histogram
data['Rating'].hist(bins=30)
plt.show()
m = data[data['Rating']<=5]
data= m
m.hist(bins=30)
plt.show()
#Code ends here


# --------------
# code starts here
total_null = data.isnull().sum()
percent_null = (total_null/data.isnull().count())
missing_data = pd.concat([total_null,percent_null],axis=1,keys=['Total','Percent'] )
print('Missing Data',missing_data)

# Cleaning- Remove Null
data=data.dropna()
total_null_1 = data.isnull().sum()
percent_null_1 = (total_null_1/data.isnull().count())
missing_data_1 = pd.concat([total_null_1,percent_null_1],axis=1,keys=['Total','Percent'] )
print('Missing Data-New',missing_data_1)
# code ends here


# --------------

#Code starts here
g = sns.catplot(x="Category",y="Rating",data=data, kind="box",height = 10)
g.set_xticklabels(rotation=90)
g.set_titles("Rating vs Category [BoxPlot]")
#Code ends here


# --------------
#Importing header files
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

#Code starts here
print(data['Installs'].value_counts)
#data['Installs'].replace(to_replace=",",value="" )
#data['Installs'].replace(to_replace="+",value="" )
data['Installs']=data['Installs'].replace('\,','',regex=True)
data['Installs']=data['Installs'].replace('\+','',regex=True)
#data['Installs'] = data['Installs'].map({',': '', '+': ''})
data['Installs']=data['Installs'].astype(int)
le = LabelEncoder()
data['Installs'] = le.fit_transform(data['Installs'])
a=sns.regplot(x="Installs", y="Rating",data=data)
a.set_title("Rating vs Installs [RegPlot]")
#Code ends here



# --------------
#Code starts here
print(data['Price'].value_counts)
data['Price']=data['Price'].replace('\$','',regex=True)
data['Price']=data['Price'].astype(float)
b=sns.regplot(x="Price", y="Rating",data=data)
b.set_title("Rating vs Price [RegPlot]")

#Code ends here


# --------------

#Code starts here
print(data['Genres'].unique())

data['Genres'] = data["Genres"].str.split(";", n = 1, expand = True)[0]
gr_mean = data[['Genres','Rating']].groupby(['Genres'],as_index=False).mean()
print(gr_mean.describe())
gr_mean = gr_mean.sort_values("Rating",axis = 0)
#gr_mean=gr_mean.apply(lambda _gr_mean: _gr_mean.sort_values(by=['Rating']))
#gr_mean = gr_mean.sort_values(by = ["Rating"],ascending = True)
print(gr_mean.iloc[0])
print(gr_mean.iloc[-1])
#Code ends here




# --------------

#Code starts here
print(data['Last Updated'])
data['Last Updated'] = data['Last Updated'].astype('datetime64[ns]')
max_date=max(data['Last Updated'])
data['Last Updated Days'] = (max_date - data['Last Updated']).dt.days

l=sns.regplot(x="Last Updated Days", y="Rating", data=data)
l.set_title("Rating vs Last Updated [RegPlot]")
#Code ends here


