# --------------
import pandas as pd
from sklearn.model_selection import train_test_split
df = pd.read_csv(path)
#print(df.head(5))
y = df['list_price'].copy()
X = df[['ages','num_reviews','piece_count','play_star_rating','review_difficulty','star_rating','theme_name','val_star_rating','country']].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=6)




# --------------
import matplotlib.pyplot as plt

# code starts here  
#print(X_train.shape)   
cols = X_train.columns
#cols.set_index("ages", inplace = True)
#print(cols.head(4))
fig ,axes = plt.subplots(nrows = 3 , ncols = 3)

for i in range(0,3):
    for j in range (0,3):
        col = cols[i* 3 +j]
        axes[i,j].scatter(X_train[col],y_train,c='b')

plt.show()

# code ends here



# --------------
# Code starts here
import numpy as np
corr = X_train.corr(method ='pearson')
print(corr)
m = ~(corr.mask(np.eye(len(corr), dtype=bool)).abs() > 0.75).any()
raw = corr.loc[m, m]
#print(raw)
X_train.drop(['play_star_rating', 'val_star_rating'], inplace = True,axis=1)
X_test.drop(['play_star_rating', 'val_star_rating'], inplace = True,axis=1)
# Code ends here


# --------------
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Code starts here
regressor = LinearRegression().fit(X_train,y_train)
y_pred = regressor.predict(X_test)
mse = mean_squared_error(y_test,y_pred)
print('mse-' ,mse)
r2 = r2_score(y_test,y_pred)
print('Rsquare',r2)


# Code ends here


# --------------
# Code starts here
er = []
residual = y_test-y_pred



print('Residual' , residual)
plt.hist(residual)
plt.show()
# Code ends her


