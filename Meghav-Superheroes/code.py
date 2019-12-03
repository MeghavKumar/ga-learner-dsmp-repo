# --------------
#code starts here
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv(path)
data['Gender']=data['Gender'].replace('-' , 'Agender')
gender_count = data['Gender'].value_counts()
print(gender_count)
data['Gender'].value_counts().plot(kind = 'bar')
#plt.bar(gender_count,data['Gender'],align='center', alpha=0.5)
plt.show()


# --------------
#Code starts here
alignment = data['Alignment'].value_counts()
print(alignment)
data['Alignment'].value_counts().plot(kind = 'pie')
plt.legend(labels= "Character Alignment")


# --------------
# Code for Strength and Combat Correlation
sc_df = data[['Strength','Combat']]
sc_covariance = sc_df['Strength'].cov(sc_df['Combat'])
print('ScCovariance',sc_covariance)
sc_strength = (sc_df['Strength'].std())
print('Sc-Strength',sc_strength)
sc_combat = (sc_df['Combat'].std())
print('Sc-Combat',sc_combat)
sc_pearson = sc_covariance /(sc_strength*sc_combat)
print('Pearson',sc_pearson)
# Code for Intelligence and Combat Correlation
ic_df = data[['Intelligence','Combat']]
ic_covariance = (ic_df['Intelligence'].cov(ic_df['Combat']))
print('IcCovariance',ic_covariance)
ic_intelligence = (ic_df['Intelligence'].std())
print('Ic-Intelligence',ic_intelligence)
ic_combat = (ic_df['Combat'].std())
print('Ic-Combat-New',ic_combat)
ic_pearson = ic_covariance /(ic_intelligence *ic_combat )
print('Ic Pearson',ic_pearson)
print('ScCovariance',sc_covariance)


# --------------
#Code starts here
total_high=(data['Total'].quantile(q=0.99))
#total_high = round(thh ,2)
#print(total_high)
super_best = data[(data['Total']>total_high)]
#print(super_best)
super_best_names =[]
super_best_names.append(super_best['Name'])
print('Superbest names',super_best_names)


# --------------
#Code starts here
fig,(ax_1,ax_2,ax_3) = plt.subplots(3)
ax_1.boxplot(data['Intelligence'])
ax_1.set_title('Intelligence')
ax_2.boxplot(data['Speed'])
ax_2.set_title('Speed')
ax_3.boxplot(data['Power'])
ax_3.set_title('Power')


