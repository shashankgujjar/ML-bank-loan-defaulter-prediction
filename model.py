# ... Necesasry Libraries ...

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import metrics 
from sklearn import neighbors
from sklearn import ensemble


# ... Data Loading ...
data = pd.read_csv("E:\\Data\\Projects\\bank fraud detection\\bank_final.csv")

# ... Conversion Data Frame ...
df = pd.DataFrame(data)
cols = df.columns

df.columns
df.shape
df.head()
df.dtypes
df.describe()
df.info()


# Normality check
df.hist()
df.skew()


# ... EDA ...

# Column Preparations

# 1... Removing unnecessary columns ...
df.drop(['Name','City','State','Zip','Bank','BankState',
         'CCSC','ApprovalDate','ChgOffDate','DisbursementDate'], inplace=True, axis=1)
df.shape

# 2... Removing Duplicates ...
# from above report we observe that many values are duplicate so we need to remove those
df.duplicated().sum()
df.shape

df.drop_duplicates(inplace=True) 
df.shape # 16,408 observations are drpped

# 3... Null values Treatment ...

# Null Percentage
nulls_percent=[]
for i in data.columns:
    tmp=len(data[data[i].isnull()])/len(data)*100
    nulls_percent.append(round(tmp,2))
null_df=pd.DataFrame(data=nulls_percent,index=data.columns,columns=['% Nulls'])
null_df[null_df['% Nulls']!=0]

# Count
df.isna().sum()

# Removing / Replacing

# ApprovalFY
df['ApprovalFY'].value_counts()
df['ApprovalFY'].unique()
df['ApprovalFY'].isna().sum()


# Term 
df['Term'].isna().sum()
df['Term'].value_counts().sum()
df['Term'].nunique()


# NoEmp
df['NoEmp'].isna().sum()
df['NoEmp'].value_counts().sum()
df['NoEmp'].nunique()


# CreateJob
df['CreateJob'].isna().sum()
df['CreateJob'].value_counts().sum()
df['CreateJob'].nunique()
df['CreateJob'].unique()


# RetainedJob
df['RetainedJob'].isna().sum()
df['RetainedJob'].value_counts().sum()
df['RetainedJob'].nunique()
df['RetainedJob'].unique()


# NewExist
df['NewExist'].value_counts()
df['NewExist'].isna().sum()

# 3 unique values are present, so conerting 0s to mode value
# and converting to binary
NewExist_Mode = df['NewExist'].mode()

df['NewExist'] = df['NewExist'].replace(0,1)
df['NewExist'].unique()
df['NewExist'].value_counts()


# FranchiseCode
# 0 & 1 represents No Franchise, so replacing 1 with 0
df['FranchiseCode'].nunique()

df['FranchiseCode'].isna().sum()
df['FranchiseCode'] = df['FranchiseCode'].replace(1,0)

# wherever code is not equal to 0, we replace it with 1, which represents Franchise
df['FranchiseCode'] = np.where((df.FranchiseCode != 0), 1,df.FranchiseCode)
df['FranchiseCode'].unique()
df['FranchiseCode'].value_counts()



# UrbanRural
df['UrbanRural'].nunique()
df['UrbanRural'].value_counts()

UrbanRural_Mode = df['UrbanRural'].mode()

# Replacing 2s with mode value i.e., 1
#df['UrbanRural'] = df['UrbanRural'].replace(0,2)

df['UrbanRural'].unique()
df['UrbanRural'].value_counts()


# RevLineCr
df['RevLineCr'].isna().sum()

df['RevLineCr'].nunique()
df['RevLineCr'].value_counts()

spec_chars = ("!",'"',"#","%","&","'","(",")",
              "*","+",",","-",".","/",":",";","<",
              "=",">","?","@","[","\\","]","^","_",
              "`","{","|","}","~","–","T","''","' '")
for char in spec_chars:
    df['RevLineCr'] = df['RevLineCr'].str.replace(char,'Y')

df['RevLineCr'].value_counts()

df['RevLineCr'].fillna('Y', inplace=True)
df['RevLineCr'].value_counts()

# 0 = No, 1 = Yes
df['RevLineCr'] = df['RevLineCr'].replace('N','0')
df['RevLineCr'] = df['RevLineCr'].replace('Y','1')

df['RevLineCr'].value_counts()

df['RevLineCr'].isna().sum()

# LowDoc
# Dummies Creation for categorical data ...
# converting Categorical data to Numerical
df['LowDoc'].value_counts()

df['LowDoc'].unique()

LowDoc_Mode = df['LowDoc'].mode()

df['LowDoc'] = df['LowDoc'].str.replace('C','N')
df['LowDoc'] = df['LowDoc'].str.replace('1', 'Y')

df['LowDoc'].value_counts()

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
le = LabelEncoder()

# No = 0, Yes = 1
df['LowDoc'] = le.fit_transform(df['LowDoc'])
df['LowDoc'].unique()

df['LowDoc'].value_counts()


# MIS_Status
# Replacing the Non Defaulter(P I F) = 1 ....
# Defaulter (CHGOFF) = 2 ...

df['MIS_Status'].isna().sum()
df['MIS_Status'].unique()

df['MIS_Status'].value_counts()

df['MIS_Status'] = df['MIS_Status'].map({'P I F':1, 'CHGOFF':2})

df.rename(columns = { "MIS_Status" : "status"}, inplace = True)

df['Status'] = df['status']

df.shape

df.drop(['status'], inplace=True, axis=1)
df.shape
# Replacing null values with mode i.e, 1
# MIS_Status_mode = df['MIS_Status'].mode()
df['Status'].value_counts()
df['Status'].unique()

df['Status'].fillna(1, inplace=True)

df['Status'].nunique()
df['Status'].unique()

df.Status.value_counts()

df['Status'].isna().sum()


# 4... Removing $ sign from currency columns 

Currency_Columns = ['DisbursementGross','BalanceGross','ChgOffPrinGr',
                    'GrAppv','SBA_Appv']

df[Currency_Columns] = df[Currency_Columns].replace('[\$,]',"",
                        regex=True).astype(float)
df.columns


# ... Method 2: PCA ...
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale 

# Considering only input data 
df.data = df.iloc[:,0:15]
df.data.head(4)

# Normalizing the numerical data 
df_normal = scale(df.data)

pca = PCA(n_components = 15)
pca_values = pca.fit_transform(df_normal)


# The amount of variance that each PCA explains is 
var = pca.explained_variance_ratio_
var
pca.components_[0]

# Cumulative variance (out of 100%)
var1 = np.cumsum(np.round(var,decimals = 2)*100)
var1

df.columns
df.shape

# Variance plot for PCA components obtained 
plt.plot(var1,color="red")

# plot between PCA1 and PCA2 
x = pca_values[:,0]
y = pca_values[:,1]
plt.scatter(x,y,color=["red"])

# PCA result sayas that first 9 columns are enough to explain the data
# so we can remove the rest

df.drop(['LowDoc','DisbursementGross', 'BalanceGross', 'ChgOffPrinGr',
         'GrAppv','SBA_Appv'], inplace = True, axis=1) 

df.columns
df.shape

# Model Building using PCA - Feature Engineering

# Defining the Inputs and Outputs
colnames = list(df.columns)
predictors = colnames[0:9]
target = colnames[9]

# Data Spliting into Train And Test 
train,test = train_test_split(df,test_size = 0.2) # 0.2 => 20 percent of entire data 


#Decision Tree Algorithm:
from sklearn.tree import  DecisionTreeClassifier as DTC

# Decision Tree Model 1 using 'entropy' index
model1=DTC(criterion = 'entropy')
model1.fit(train[predictors],train[target])
preds = model1.predict(test[predictors])

# Counting the Value of Non-Default and Default
pd.Series(preds).value_counts()

# just like confusion matrix
pd.crosstab(test[target],preds)


print("Accuracy",(17853+6365)/(1100+1392+17853+6365)*100)
# 90.67016098839386 

# Train Accuracy
np.mean(train.Status == model1.predict(train[predictors])) 
# 97.19851785313272

# Test Accuracy
np.mean(test.Status == model1.predict(test[predictors])) 
# 90.67016098839386 


## Decision Tree Model 2 using 'gini' index
#model2=DTC(criterion = 'gini')
#model2.fit(train[predictors],train[target]) 
#preds = model2.predict(test[predictors])
#
## Counting the Value of Non-Default and Default
#pd.Series(preds).value_counts()
#
## just like confusion matrix
#pd.crosstab(test[target],preds)
#
#print("Accuracy",(17852+6359)/(1398+1110+17852+6359)*100)
## 90.6134211609716
#
## Train Accuracy
#np.mean(train.Status == model2.predict(train[predictors])) 
## 97.19851785313272
#
## Test Accuracy
#np.mean(test.Status == model2.predict(test[predictors])) 
## 90.6134211609716



# The ENTROPY model shows a better accuracy than ENTROPY model with medium fit of train accuracy
# Hence we can consider the model as final



# ... Model Evaluation ...

# Method : Bagging... 

from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier

bg = BaggingClassifier(DTC(), max_samples=0.5,max_features=1.0, n_estimators=2000)

model = bg.fit(train[predictors],train[target])

bg.score(test[predictors],test[target]) #test accuracy
# 92.59328567685916

bg.score(train[predictors],train[target]) #train accuracy 
# 96.0878434014522

# the above scores are generalized
# so we select the above model as final


# Deployment

import pickle

pickle.dump(bg, open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))

