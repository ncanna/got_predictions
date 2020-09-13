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

### K-Nearest Neighbors (KNN)

KNN models were run on unscaled and standard scaled data, with the latter performing significantly better. The unscaled model gave a *testing accuracy of 0.74* while the scaled model gave a *testing accuracy of 0.969*. Both models were evaluated using the number of neighbors that yielded the best test accuracy. In this case, between a range of 1 and 51 neighbors, 7 neighbors was ideal for both models.

### Gradient Boosting Classifier (GBC)

The GBC model used the friedman_mse criterion and a deviance loss function. In each stage ```n_classes_``` regression trees are fit on the negative gradient of the binomial or multinomial deviance loss function. 100 estimators were used to give a *testing accuracy of 0.964*. 

### Random Forest, optimized using RandomizedSearchCV

A tree model was optimized via RandomizedSearchCV. Both entropy (with a *testing accuracy of 0.956*) and gini (also with a *testing accuracy of 0.956*) criterion were tested and evaluated using roc_auc scores. 

### Feauture Importance

Using the Random Forest model, feauture importances were measured and plotted. The results showed that of the factors considered, a character's house death rate and what season number they were introduced in were most important when predicting if they would survive the show or not.

!(https://miro.medium.com/max/1400/1*ABimYBeJVJerGFM06OBiZw.png)
