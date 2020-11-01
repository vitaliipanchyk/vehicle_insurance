import numpy as np
import pandas as pd


# preprocessing
from sklearn.model_selection import train_test_split


# models
from sklearn.linear_model import LinearRegression,LogisticRegression,  RidgeClassifier, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

#loading data
data = pd.read_csv("6.Train.csv")

#feature engineering
data = data.drop("id", axis=1)
data = data.drop("Region_Code", axis=1)
data["Genders"] = data.Gender.apply(lambda x: 0  if x == "Male" else 1)
data =  data.drop("Gender",axis=1)
data.replace({"< 1 Year":0, "1-2 Year":1, "> 2 Years":2}, inplace=True)
data.Vehicle_Damage.replace({"Yes": 1, "No":0}, inplace=True)
data.drop_duplicates(inplace=True)

#LOG TRANSFORM
data['AP_log'] = (data['Annual_Premium']+1).transform(np.log)
data = data.drop("Annual_Premium",axis=1)
data = data.drop("Policy_Sales_Channel", axis=1)
data['Vintage_log'] = (data['Vintage']+1).transform(np.log)
data=data.drop("Vintage", axis=1)
data['Age_log'] = (data['Age']+1).transform(np.log)
data=data.drop("Age", axis=1)

#undersampling
positive_data = data[data["Response"] == 1]
negative_data = data[data["Response"] == 0]
short_negative_data = negative_data.iloc[:63000,]
prepared_data = pd.concat([positive_data, short_negative_data])
print(prepared_data.Response.value_counts())



features = prepared_data.drop("Response",axis=1)
target = prepared_data["Response"]


#train_test_split
features_train, features_val, targets_train, targets_val = train_test_split(features, target, test_size=0.2, random_state=12)

#building and evaluating models
LogReg = LogisticRegression()
LogReg.fit(features_train, targets_train)
acc_log_reg_train = round(LogReg.score(features_train, targets_train) * 100, 2)
acc_log_reg_val = round(LogReg.score(features_val, targets_val) * 100, 2)
print("LogReg train " + str(acc_log_reg_train))
print("LogReg val " + str(acc_log_reg_val))
guesses = LogReg.predict(features_val)
f1_score = metrics.f1_score(targets_val, guesses)
print("LogReg f1 " + str(f1_score))

gaussian = GaussianNB()
gaussian.fit(features_train, targets_train)
acc_gaussian_train = round(gaussian.score(features_train, targets_train) * 100, 2)
acc_gaussian_val = round(gaussian.score(features_val, targets_val) * 100, 2)
print("NB train " + str(acc_gaussian_train))
print("NB val " + str(acc_gaussian_val))
guesses1 = gaussian.predict(features_val)
f1_score = metrics.f1_score(targets_val, guesses1)
print("NB f1 " + str(f1_score))

decision_tree = DecisionTreeClassifier()
decision_tree.fit(features_train, targets_train)
acc_decision_tree_train = round(decision_tree.score(features_train, targets_train) * 100, 2)
acc_decision_tree_val = round(decision_tree.score(features_val, targets_val) * 100, 2)
print("tree train" + str(acc_decision_tree_train))
print("tree val" + str(acc_decision_tree_val))
guesses2 = decision_tree.predict(features_val)
f1_scoreDT = metrics.f1_score(targets_val, guesses2)
print("tree " + str(f1_scoreDT))

KNC = KNeighborsClassifier(n_neighbors=2)
KNC.fit(features_train, targets_train)
acc_KNC_train = round(KNC.score(features_train, targets_train) * 100, 2)
acc_KNC_val = round(KNC.score(features_val, targets_val) * 100, 2)
print("KNC train" + str(acc_KNC_train))
print("KNC val" + str(acc_KNC_val))
guesses3 = KNC.predict(features_val)
f1_scoreKNC = metrics.f1_score(targets_val, guesses3)
print(f1_scoreKNC)

sgd = SGDClassifier()
sgd.fit(features_train, targets_train)
acc_sgd_train = round(sgd.score(features_train, targets_train) * 100, 2)
acc_sgd_val = round(sgd.score(features_val, targets_val) * 100, 2)
print("SGD train" + str(acc_sgd_train))
print("SGD val" + str(acc_sgd_val))
guesses4= sgd.predict(features_val)
f1_scoreSGD = metrics.f1_score(targets_val, guesses4)
print(f1_scoreSGD)

ridge_classifier = RidgeClassifier()
ridge_classifier.fit(features_train, targets_train)
acc_ridge_train = round(ridge_classifier.score(features_train, targets_train) * 100, 2)
acc_ridge_val = round(ridge_classifier.score(features_val, targets_val) * 100, 2)
print("Ridge train" + str(acc_ridge_train))
print("Ridge val" + str(acc_ridge_val))
guesses5= ridge_classifier.predict(features_val)
f1_score_ridge = metrics.f1_score(targets_val, guesses5)
print(f1_score_ridge)


