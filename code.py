# --------------
# Loading the Necessary Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load the dataset

df = pd.read_csv(path)
df.head()
# Check the correlation between each feature and check for null values

corr = df.corr()
sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns)
df.isna().sum()
# Print total no of labels also print number of Male and Female labels
print('Total',len(df))

df.label.value_counts()

# Label Encode target variable
X = df.drop('label',1)
y = df.label

gender_encoder = LabelEncoder()

y = gender_encoder.fit_transform(y)
# Scale all the independent features and split the dataset into training and testing set.
scaler = StandardScaler()
scaler.fit(X)

X = scaler.transform(X)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=1)


# Build model with SVC classifier keeping default Linear kernel and calculate accuracy score.
svc = SVC(kernel = 'linear')
svc.fit(X_train,y_train)

y_pred = svc.predict(X_test)

print(accuracy_score(y_test,y_pred))

# Build SVC classifier model with polynomial kernel and calculate accuracy score
svc_poly = SVC(kernel='poly')

svc_poly.fit(X_train,y_train)

y_pred_poly = svc_poly.predict(X_test)

print(accuracy_score(y_test,y_pred_poly))

# Build SVM model with rbf kernel.
svc_rbf = SVC(kernel='rbf')

svc_rbf.fit(X_train,y_train)

y_pred_rbf = svc_rbf.predict(X_test)

print(accuracy_score(y_test,y_pred_rbf))

#  Remove Correlated Features.
corr_matrix = df.drop('label',1).corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape),k=1).astype(np.bool))

to_drop = [column for column in upper.columns if any(upper[column]>0.95)]
print(to_drop)

df.drop(to_drop,1,inplace=True)

# Split the newly created data frame into train and test set, scale the features and apply SVM model with rbf kernel to newly created dataframe

X = df.drop('label',1)
y = df.label

gen_encoder = LabelEncoder()

gen_encoder.fit_transform(y)

scaler = StandardScaler()

scaler.fit(X)

X = scaler.transform(X)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state = 1)

svc_rb1 = SVC(kernel = 'rbf')

svc_rb1.fit(X_train,y_train)

y_pred1 = svc_rb1.predict(X_test)

print(accuracy_score(y_test,y_pred1))


# Do Hyperparameter Tuning using GridSearchCV and evaluate the model on test data.
param_dict = {'C':[0.001,0.01,0.1,1,10,100],
                'gamma':[0.001,0.01,0.1,1,10,100],
                'kernel':['linear','rbf']}

svc_cv = GridSearchCV(estimator = SVC(),param_grid = param_dict ,scoring = 'accuracy',cv=10)
svc_cv.fit(X_train,y_train)

y_pred12 = svc_cv.predict(X_test)

print(accuracy_score(y_test,y_pred12))


