# -*- coding: utf-8 -*-

# Imports
"""

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import cross_val_score

#Loading the file
preds = pd.read_csv('got.csv')
preds.head(5)

preds.info

"""# Exploratory Analysis"""

#Number of dead vs. alive characters
equiv = {1:"Alive", 0:"Dead"}
preds["isAlive_str"] = preds["isAlive"].map(equiv)
fig = px.pie(preds, names="isAlive_str", title="Dead vs. Alive Characters")
fig.update_traces(textposition='inside', textinfo='percent+label')
fig.show()

#Characters that do not appear in any books
preds['total_books_in'] = (preds['book1'] + 
                   preds['book2'] +
                   preds['book3'] +
                   preds['book4']+
                   preds['book5'])

#Number books characters appear in
fig = px.pie(preds, names="total_books_in", title="Number books characters appear in")
fig.update_traces(textposition='inside', textinfo='label+percent')
fig.show()

showonly_str = {0:"Yes", 1:"No", 2:"No", 3:"No", 4:"No", 5:"No", 6:"No"}
showonly_num = {0:1, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0}
preds["showOnly"] = preds["total_books_in"].map(showonly_num)
preds["showOnly_str"] = preds["total_books_in"].map(showonly_str)
preds["showOnlyDead"]=np.where(np.logical_or(preds['showOnly']==1,preds['isAlive']==0),1,0)
preds.head(5)

preds['showOnly'].value_counts()

fig = px.pie(preds, values='showOnly', names='isAlive_str', title="Percentage of Show Only Characters Alive")
fig.update_traces(textposition='inside', textinfo='label+percent')
fig.show()

fig = px.pie(preds, names='isAlive_str', title="Percentage of All Characters Alive")
fig.update_traces(textposition='inside', textinfo='label+percent')
fig.show()

#Exploring age 
preds['age'].value_counts()
fig = px.scatter(preds, x="age", title="Ages of All Characters")
fig.show()

#Exploring DOB
preds['dateOfBirth'].value_counts()
fig = px.scatter(preds, x="dateOfBirth", title="Dates of Birth of Characters")
fig.show()

#Fixing the outlier age and date of births manually 
preds['age'].values[preds['age'] == -298001] = 0
preds['age'].values[preds['age'] == -277980] = 20
preds['dateOfBirth'].values[preds['dateOfBirth'] == 298299] = 298
preds['dateOfBirth'].values[preds['dateOfBirth'] == 278279] = 278

#Exploring age
preds['age'].value_counts()
fig = px.scatter(preds, x="age", title="Cleaned Ages of All Characters")
fig.show()

#Exploring DOB
preds['dateOfBirth'].value_counts()
fig = px.scatter(preds, x="dateOfBirth", title="Dates of Birth of Characters")
fig.show()

#Does popularity cause death?
preds_dead = preds[preds.isAlive == 0]
fig = px.histogram(preds_dead, x="popularity", title="Popularity of Dead Characters")
fig.show()

#Popularity of alive characters
preds_alive = preds[preds.isAlive == 1]
fig = px.histogram(preds_alive, x="popularity", title="Popularity of Alive Characters")
fig.show()

#Popularity of show only characters
preds_so = preds[preds.showOnly == 1]
fig = px.histogram(preds_so, x="popularity", title="Popularity of Alive Characters")
fig.show()

#Does gender affect death?
fig = px.histogram(preds, x="isAlive", color="male", title="Gender of Alive and Dead Characters")
#fig = px.histogram(preds, x="isAlive", color="male", title="Gender of Alive and Dead Characters", barnorm='percent')
fig.show()

#Gender Distribution
male_str = {0:"Female", 1:"Male"}
preds["male_str"] = preds["male"].map(male_str)
fig = px.pie(preds, names="male_str", title="Gender of All Characters")
fig.update_traces(textposition='inside', textinfo='label+percent')
fig.show()

#Gender Distribution
preds_dead = preds[preds.isAlive == 0]
fig = px.pie(preds_dead, names="male_str", title="Gender of Dead Characters")
fig.update_traces(textposition='inside', textinfo='label+percent')
fig.show()

#Does number of dead relations affect death?
fig = px.histogram(preds, x="numDeadRelations", color="isAlive", title="Number Dead Relations Alive and Dead Characters")
fig.show()

#Does nobility affect death?
fig = px.histogram(preds, x="isAlive_str", color="isNoble", title="Nobility of Alive and Dead Characters")
fig.show()

#Imputing median for Date of Birth and Age
dob_median = preds['dateOfBirth'].median()
preds['dateOfBirth'] = preds['dateOfBirth'].fillna(dob_median).round(3)
age_median = preds['age'].median()
preds['age'] = preds['age'].fillna(age_median).round(3)

#Calculating house danger
preds['houseSize'] = preds['house'].map(preds['house'].value_counts())
preds['houseAlive'] = preds_alive['house'].map(preds_alive['house'].value_counts())
preds['houseDead'] = preds_dead['house'].map(preds_dead['house'].value_counts())
preds['houseDeathRate'] = preds['houseDead']/preds['houseSize']

#Imputing other numerical variables with -1 (to indicate being missing)
num_fill = ['isAliveMother', 'isAliveFather', 'isAliveHeir', 'isAliveSpouse', 'houseSize', 'houseAlive', 'houseDead', 'houseDeathRate']

for col in preds[num_fill]:
    if preds[col].isnull().astype(int).sum() > 0:
        preds[col] = preds[col].fillna(-1)

#Imputing categorical missing values with unknown
cat_fill =['title', 'culture','mother','father', 'heir', 'house', 'spouse']

for col in preds[cat_fill]:
    if preds[col].isnull().astype(int).sum() > 0:
        preds[col] = preds[col].fillna('unknown')

#Are some houses more dangerous?
fig = px.histogram(preds, x="isAlive_str", color="house", title="Houses of Alive and Dead Characters")
fig.show()

#Plotting house death rates
fig = px.histogram(preds, x="house", y="houseDeathRate", title="Nobility of Alive and Dead Characters")
fig.show()

#Checking for missing values
preds.isnull().sum()

"""# Models"""

abs(preds.corr()['isAlive']) > 0.15

preds.corr().round(3)

preds_partial = preds[['S.No',
                      'male',
                      'dateOfBirth',
                      'book1',
                      'book4',
                      'numDeadRelations',
                      'boolDeadRelations',
                      'popularity',
                      'showOnly',
                      'houseDeathRate']]

preds_target = preds.loc[: ,'isAlive']

"""## Logistic Regression, sklearn"""

from sklearn.linear_model import LogisticRegression

#Train, test, split
X_train, X_test, y_train, y_test = train_test_split(preds_partial,
                                                    preds_target,
                                                    test_size = 0.2,
                                                    random_state = 508)

#Instantiate and fit
lr = LogisticRegression()
lr_fit = lr.fit(X_train, y_train)

#Predict
lr_pred = lr_fit.predict(X_test)
y_score_ols = lr_fit.score(X_test, y_test)

print('Model Score', y_score_ols) 
print('Training Score', lr_fit.score(X_train, y_train).round(7))
print('Testing Score:', lr_fit.score(X_test, y_test).round(7))

"""## Decision Tree"""

from sklearn.tree import DecisionTreeClassifier

#Train, test, split
X_train, X_test, y_train, y_test = train_test_split(
            preds_partial,
            preds_target,
            test_size = 0.2,
            random_state = 508,
            stratify = preds_target)

#Instantiate and fit
c_tree = DecisionTreeClassifier(random_state = 508)
c_tree_fit = c_tree.fit(X_train, y_train)

#Printing model scores
print('Training Score', c_tree_fit.score(X_train, y_train).round(7))
print('Testing Score:', c_tree_fit.score(X_test, y_test).round(7))

"""## KNN Models

### Unscaled KNN
"""

from sklearn.neighbors import KNeighborsClassifier 

#Train, test, split
X_train, X_test, y_train, y_test = train_test_split(
            preds_partial,
            preds_target,
            test_size = 0.2,
            random_state = 508)

training_accuracy = []
test_accuracy = []

#Instantiate and fit
neighbors_settings = range(1, 51)
for n_neighbors in neighbors_settings:
    clf = KNeighborsClassifier(n_neighbors = n_neighbors)
    clf.fit(X_train, y_train)
    training_accuracy.append(clf.score(X_train, y_train))
    test_accuracy.append(clf.score(X_test, y_test))

#Plot training and test accuracies
plt.plot(neighbors_settings, training_accuracy, label = "training accuracy")
plt.plot(neighbors_settings, test_accuracy, label = "test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()

#Find optimal number neighbors via test accuracy
optimal_neighbors = test_accuracy.index(max(test_accuracy)) + 1
print('Highest test accuracy was at {} neighbors. '.format(optimal_neighbors))

#Instantiate and fit
knn_reg = KNeighborsClassifier(algorithm = 'auto', n_neighbors = 3)
knn_reg_fit = knn_reg.fit(X_train, y_train)
y_score_knn_train = knn_reg.score(X_train, y_train)
y_score_knn_optimal = knn_reg.score(X_test, y_test)

#Printing model scores
cv = cross_val_score(knn_reg, preds_partial, preds_target, cv = 3, 
                          scoring =  'roc_auc')
print('Training Score', y_score_knn_train.round(4))
print('Testing Score:', y_score_knn_optimal.round(4))
print('AUC Score:', pd.np.mean(cv))

"""### Scaled KNN"""

from sklearn.preprocessing import StandardScaler

#Instantiate scaler and add labels
scaler = StandardScaler()
scaler.fit(preds_partial)
preds_scaled = scaler.transform(preds_partial)
preds_scaled_table = pd.DataFrame(preds_scaled)
preds_scaled_table.columns = preds_partial.columns

#Train, test, split with scaled data
X_train, X_test, y_train, y_test = train_test_split(
            preds_scaled_table,
            preds_target,
            test_size = 0.2,
            random_state = 508)

training_accuracy = []
test_accuracy = []

#Instantiate and fit
neighbors_settings = range(1, 51)
for n_neighbors in neighbors_settings:
    clf = KNeighborsClassifier(n_neighbors = n_neighbors)
    clf.fit(X_train, y_train)
    training_accuracy.append(clf.score(X_train, y_train))
    test_accuracy.append(clf.score(X_test, y_test))

plt.plot(neighbors_settings, training_accuracy, label = "training accuracy")
plt.plot(neighbors_settings, test_accuracy, label = "test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()

#Find optimal number neighbors via test accuracy
print(test_accuracy.index(max(test_accuracy)) + 1)
knn_reg = KNeighborsClassifier(algorithm = 'auto',
                              n_neighbors = 27)

#Instantiate and fit
knn_reg_fit = knn_reg.fit(X_train, y_train)
y_score_knn_train = knn_reg.score(X_train, y_train)
y_score_knn_optimal = knn_reg.score(X_test, y_test)


#Printing model scores
cv = cross_val_score(knn_reg,preds_partial, preds_target, cv = 3, 
                          scoring =  'roc_auc')
print('Training Score', y_score_knn_train.round(4))
print('Testing Score:', y_score_knn_optimal.round(4))
print('AUC Score:', pd.np.mean(cv))

"""## GBC"""

from sklearn.ensemble import GradientBoostingClassifier

#Train, test, split
X_train, X_test, y_train, y_test = train_test_split(
            preds_partial,
            preds_target.values.ravel(),
            test_size = 0.1,
            random_state = 508,
            stratify = preds_target)

#Instantiate and fit
gbm = GradientBoostingClassifier(loss = 'deviance',
                                  learning_rate = 1.5,
                                  n_estimators = 100,
                                  max_depth = 3,
                                  criterion = 'friedman_mse',
                                  warm_start = False,
                                  random_state = 508,
                                  )
gbm_basic_fit = gbm.fit(X_train, y_train)
gbm_basic_predict = gbm_basic_fit.predict(X_test)

#Printing model scores
cv = cross_val_score(gbm, preds_partial, preds_target, cv = 3, 
                          scoring =  'roc_auc')
print('Training Score', gbm_basic_fit.score(X_train, y_train).round(4))
print('Testing Score:', gbm_basic_fit.score(X_test, y_test).round(4))
print('AUC Score:', pd.np.mean(cv))

"""## Random Forest Models"""

from sklearn.ensemble import RandomForestClassifier

#Train, test, split
X_train, X_test, y_train, y_test = train_test_split(
            preds_partial,
            preds_target.values.ravel(),
            test_size = 0.2,
            random_state = 508,
            stratify = preds_target)

#Instantiate and fit Gini
RF_gini = RandomForestClassifier(n_estimators = 100,
                                     criterion = 'gini',
                                     max_depth = None,
                                     min_samples_leaf = 1,
                                     bootstrap = False,
                                     warm_start = False,
                                     random_state = 508)

#Fitting model
RF_gini_fit = RF_gini.fit(X_train, y_train)

#Printing model scores
print('Training Score', RF_gini_fit.score(X_train, y_train).round(7))
print('Testing Score:', RF_gini_fit.score(X_test, y_test).round(7))

#Instantiate and fit Entropy
RF_entropy = RandomForestClassifier(n_estimators = 100,
                                     criterion = 'entropy',
                                     max_depth = None,
                                     min_samples_leaf = 1,
                                     bootstrap = False,
                                     warm_start = False,
                                     random_state = 508)
RF_entropy_fit = RF_entropy.fit(X_train, y_train)

#Printing model scores
print('Training Score', RF_entropy_fit.score(X_train, y_train).round(7))
print('Testing Score:', RF_entropy_fit.score(X_test, y_test).round(7))

#Comparing Gini and Entropy AUC scores
cv_gini = cross_val_score(RF_gini, preds_partial, preds_target, cv = 3, 
                          scoring =  'roc_auc')
print('Gini AUC Score', pd.np.mean(cv_gini))

cv_entropy = cross_val_score(RF_entropy, preds_partial, preds_target, cv = 3, 
                          scoring =  'roc_auc')
print('Entropy AUC Score', pd.np.mean(cv_entropy))

"""### RandomizedSearchCV"""

from sklearn.model_selection import RandomizedSearchCV

#Define search parameters
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
min_samples_split = [2, 4, 6, 8, 10]
min_samples_leaf = [1, 2, 3, 4]
bootstrap = [True, False]

param_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

random_grid = RandomForestClassifier(random_state = 508)

#Instantiate and fit
random_grid_cv = RandomizedSearchCV(random_grid, param_grid, cv = 3)
random_grid_cv.fit(X_train, y_train)

#Printing model scores
print("Optimal RF Parameters:", random_grid_cv.best_params_)
print("Optimal RF Accuracy:", random_grid_cv.best_score_.round(4))

"""### Optimized Random Forest"""

#Train, test, split
X_train, X_test, y_train, y_test = train_test_split(
            preds_partial,
            preds_target.values.ravel(),
            test_size = 0.2,
            random_state = 508,
            stratify = preds_target)

#Instantiate and fit optimized Gini
RF_gini = RandomForestClassifier(n_estimators = 1000,
                                     criterion = 'gini',
                                     max_depth = 50,
                                     min_samples_leaf = 4,
                                     bootstrap = True,
                                     warm_start = False,
                                     random_state = 508)

#Fitting model
RF_fit = RF_gini.fit(X_train, y_train)
RF_pred = RF_entropy.predict(X_test)

#Scoring the gini model
cv_gini = cross_val_score(RF_gini, preds_partial, preds_target, cv = 3, 
                          scoring =  'roc_auc')
print('Training Score', RF_gini_fit.score(X_train, y_train).round(7))
print('Testing Score:', RF_gini_fit.score(X_test, y_test).round(7))
print('Gini AUC Score', pd.np.mean(cv_gini))

#Instantiate and fit optimized Entropy
RF_entropy = RandomForestClassifier(n_estimators = 1000,
                                     criterion = 'entropy',
                                     max_depth = 50,
                                     min_samples_leaf = 4,
                                     bootstrap = True,
                                     warm_start = False,
                                     random_state = 508)

#Fitting model
RF_entropy_fit = RF_entropy.fit(X_train, y_train)
RF_entropy_pred = RF_entropy.predict(X_test)

#Printing model scores
cv_entropy = cross_val_score(RF_entropy, preds_partial, preds_target, cv = 3, 
                          scoring =  'roc_auc')
print('Training Score', RF_entropy_fit.score(X_train, y_train).round(7))
print('Testing Score:', RF_entropy_fit.score(X_test, y_test).round(7))
print('Entropy AUC Score', pd.np.mean(cv_entropy))

"""## Feature Importance"""

def plot_feature_importances(model, train = X_train):
    n_features = X_train.shape[1]
    fig = px.bar(X_train, model.feature_importances_, y=X_train.columns, orientation='h')
    fig.update_layout(xaxis=dict(title="Feature Importance"), yaxis=dict(title="Feature")) 
    fig.show()
        
plot_feature_importances(RF_entropy_fit, train = X_train)

"""## Final Model"""

#Instantiate and fit optimized Gini
RF_final = RandomForestClassifier(n_estimators = 1000,
                                     criterion = 'gini',
                                     max_depth = 50,
                                     min_samples_leaf = 4,
                                     bootstrap = True,
                                     warm_start = False,
                                     random_state = 508)

#Fitting model
RF_final_fit = RF_final.fit(X_train, y_train)
RF_final_pred = RF_final.predict(X_test)

#Scoring the gini model
cv_final = cross_val_score(RF_final, preds_partial, preds_target, cv = 3, 
                          scoring =  'roc_auc')
print('Training Score', RF_final_fit.score(X_train, y_train).round(7))
print('Testing Score:', RF_final_fit.score(X_test, y_test).round(7))
print('Entropy AUC Score', pd.np.mean(cv_final))

#Compiling model results and sorting by popularity
compare_df = pd.DataFrame({'Actual' : y_test,
                                     'RF_Predicted': RF_gini_pred})
model_results = pd.merge(preds, compare_df, how = 'left', left_index = True, right_index = True)
model_results = model_results.dropna(subset = ['RF_Predicted'])
model_results = model_results.sort_values(by=['popularity'], ascending=False)
model_results

#Getting entries that were correctly predicted and sorting by popularity
correct_preds = model_results.loc[model_results['Actual'] == model_results['RF_Predicted']]
correct_preds = correct_preds.sort_values(by=['popularity'], ascending=False)
correct_preds = correct_preds[['name', 'Actual','RF_Predicted']]
correct_preds.head(10)

#Getting entries that were incorrectly predicted and sorting by popularity
incorrect_preds = model_results.loc[model_results['Actual'] != model_results['RF_Predicted']]
incorrect_preds = incorrect_preds.sort_values(by=['popularity'], ascending=False)
incorrect_preds = incorrect_preds[['name', 'Actual','RF_Predicted']]
incorrect_preds.head(15)