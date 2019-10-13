import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
#loading train dataset
dataset=pd.read_csv('train (4).csv')

#loading test dataset
test=pd.read_csv('test (4).csv')

id=test['id'].values

sns.heatmap(dataset.isnull(),yticklabels=False,cbar=False,cmap='viridis')

#VISUALIZE THIS FOR ALL THE FEATURES
sns.regplot(x=dataset['theta1'],y=dataset['target'])


dataset=dataset.drop(['Unnamed: 0','index','id'],axis=1)
test=test.drop(['Unnamed: 0','index','id'],axis=1)


for i in dataset.columns:
    print(dataset[i].describe())

     
#finding outlier via iqr
sns.boxplot(x=dataset['target'])
#many outliers are there
Q1 = dataset.quantile(0.25)
Q3 = dataset.quantile(0.75)
IQR = Q3 - Q1
print(IQR)
s=(dataset < (Q1 - 1.5 * IQR))| (dataset > (Q3 + 1.5 * IQR))

dataset2 = dataset[~((dataset < (Q1 - 1.5 * IQR)) |(dataset > (Q3 + 1.5 * IQR))).any(axis=1)]
sns.boxplot(x=dataset2['target'])
 

y_train=dataset2['target'].values
X_train=dataset2.drop('target',axis=1)


from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
test = sc_X.transform(test)



from sklearn.metrics import r2_score, mean_squared_error

from sklearn.ensemble import RandomForestRegressor
forest_model=RandomForestRegressor()


from sklearn.model_selection import GridSearchCV

param_grid = { 
    'n_estimators': [100,200],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8,9],
    'min_samples_split' : [2,3,4,5,6],
    
}
CV_rfc = GridSearchCV(estimator=forest_model, param_grid=param_grid, cv= 5)
CV_rfc.fit(X_train, y_train)
y_pred=CV_rfc.predict(test)
CV_rfc.score(X_train, y_train)

# The mean squared error
print("Mean squared error: %.2f"% mean_squared_error(y_train, CV_rfc.predict(X_train)))



#creating dataframe
id=id.reshape(1000,1)
y_pred=y_pred.reshape(1000,1)
y_pred=np.concatenate((id,y_pred),axis=1)
df = pd.DataFrame(y_pred,index=range(0,1000))
df.to_csv('cerebo.csv')







