# Summary

**Goal**: 
To evaluate various machine learning classifiers (```Logistic Regression```, ```Decision Tree```, ```K-Nearest Neighbors```, ```Gradient boosting Classifier```, and ```Random Forest```) in their success in predicting the deaths of Game of Thrones characters using data from the book series.

You can find a Medium post I wrote summarizing this project at this link: https://towardsdatascience.com/predicting-who-will-die-in-game-of-thrones-using-machine-learning-6c41c0ba897e. 

## Dataset

Web scraped data from a A Wiki of Ice and Fire (https://www.kaggle.com/mylesoneill/game-of-thrones), which consists of information on approximately 2,000 show characters. Feautures include ```name```, ```gender```, ```date of birth```, ```house```, 
and even ```popularity```.

Preliminary data exploration revealed that being ```Male``` and being a member of ```House Frey```, the ```Night's Watch```, or an ```Unknown House``` significantly raised the chances of a character dying in the show.

### Preprocessing

Outliers were found for the ```age``` and ```date of birth``` fields and were manually fixed. Missing  ```date of birth``` values were imputed with the median while categorical values had missing values imputed with an 'Unknown' string.

## Models
### Logistic Regression

A logistic regression model was used to serve as a baseline classifier.

### Decision Tree

A tree model was optimized via RandomizedSearchCV.

### K-Nearest Neighbors (KNN)

KNN models were run on unscaled and standard scaled data, with the latter performing significantly better.

### Gradient Boosting Classifier (GBC)

The GBC model used the friedman_mse criterion and a deviance loss function.

### Random Forest

Both entropy and gini criterion were tested and evaluated using roc_auc scores.
