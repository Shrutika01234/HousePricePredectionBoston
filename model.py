import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
boston=load_boston()
boston_df = load_boston()
print(boston_df.keys())
print(boston_df.DESCR)

# Preparing The datsets 

dataset = pd.DataFrame(boston_df.data,columns=boston_df.feature_names)
#print(dataset)

dataset['Price'] = boston_df.target
print(dataset.head(5))

print(dataset.info())

# Summerize the data 
print(dataset.describe())
print(dataset.isnull().sum())


# EDA 
# Correlation 

print(dataset.corr())

plt.scatter(dataset['CRIM'],dataset['Price'])
plt.xlabel("Crime Rate")
plt.ylabel("Price")

plt.scatter(dataset['RM'],dataset['Price'])
plt.xlabel("RM")
plt.ylabel("Price")

import seaborn as sns
sns.regplot(x="RM",y="Price",data=dataset)

sns.regplot(x="LSTAT",y="Price",data=dataset)

sns.regplot(x="CHAS",y="Price",data=dataset)

sns.regplot(x="PTRATIO",y="Price",data=dataset)

## Independent and Dependent features

X=dataset.iloc[:,:-1]
y=dataset.iloc[:,-1]
X.head()

##Train Test Split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)

## Standardize the dataset
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()

X_train=scaler.fit_transform(X_train)

X_test=scaler.transform(X_test)

import pickle
pickle.dump(scaler,open('scaling.pkl','wb'))

from sklearn.linear_model import LinearRegression

regression=LinearRegression()

regression.fit(X_train,y_train)

print(regression.coef_)

print(regression.intercept_)

regression.get_params()

### Prediction With Test Data
reg_pred=regression.predict(X_test)

plt.scatter(y_test,reg_pred)

residuals=y_test-reg_pred

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

print(mean_absolute_error(y_test,reg_pred))
print(mean_squared_error(y_test,reg_pred))
print(np.sqrt(mean_squared_error(y_test,reg_pred)))

from sklearn.metrics import r2_score
score=r2_score(y_test,reg_pred)
print(score)

import pickle

pickle.dump(regression,open('regmodel.pkl','wb'))
pickled_model=pickle.load(open('regmodel.pkl','rb'))

## Prediction
print(pickled_model.predict(scaler.transform(boston.data[0].reshape(1,-1))))