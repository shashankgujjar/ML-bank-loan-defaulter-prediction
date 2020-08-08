# ... Necesasry Libraries ...

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split


# ... Data Loading ...
data = pd.read_csv("C:\\Users\\user\\Downloads\\deploy\\bankloan.csv")

# ... Conversion Data Frame ...
df = pd.DataFrame(data)
cols = df.columns



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


# ... Model Evaluation ...

# Method : Bagging... 

from sklearn.ensemble import BaggingClassifier

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

pickle.dump(bg, open('modelnew.pkl','wb'))
model = pickle.load(open('modelnew.pkl','rb'))

