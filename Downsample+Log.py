import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn  as sns


# preprocessing

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
import pandas_profiling as pp

# models
from sklearn.linear_model import LinearRegression,LogisticRegression, Perceptron, RidgeClassifier, SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

data = pd.read_csv("train.csv")
data = data.drop("id", axis=1)
data["Genders"] = data.Gender.apply(lambda x: 0  if x == "Male" else 1)
data =  data.drop("Gender",axis=1)
data.replace({"< 1 Year":0, "1-2 Year":1, "> 2 Years":2}, inplace=True)
data.Vehicle_Damage.replace({"Yes": 1, "No":0}, inplace=True)
data.drop_duplicates(inplace=True)

#LOG TRANSFORM
data['AP_log'] = (data['Annual_Premium']+1).transform(np.log)
data = data.drop("Annual_Premium",axis=1)
data = data.drop("Policy_Sales_Channel", axis=1)
data = data.drop("Region_Code", axis=1)
data['Vintage_log'] = (data['Vintage']+1).transform(np.log)
data=data.drop("Vintage", axis=1)
data['Age_log'] = (data['Age']+1).transform(np.log)
data=data.drop("Age", axis=1)

positive_data = data[data["Response"] == 1]
negative_data = data[data["Response"] == 0]
short_negative_data = negative_data.iloc[:63000,]
prepared_data = pd.concat([positive_data, short_negative_data])
print(prepared_data.Response.value_counts())


features = prepared_data.drop("Response",axis=1)
target = prepared_data["Response"]

#profile = pp.ProfileReport(data)
#profile.to_file(output_file="report2.html")

features_train, features_val, targets_train, targets_val = train_test_split(features, target, test_size=0.2, random_state=12)


LogReg = LogisticRegression()
LogReg.fit(features_train, targets_train)
acc_log_reg_train = round(LogReg.score(features_train, targets_train) * 100, 2)
acc_log_reg_val = round(LogReg.score(features_val, targets_val) * 100, 2)
print(acc_log_reg_train)
print(acc_log_reg_val)

guesses = LogReg.predict(features_val)
f1_score = metrics.f1_score(targets_val, guesses)
print(f1_score)

report = metrics.classification_report(targets_val, guesses, target_names=["Not Buy", "Buy"])
print(report)

