# Problem Statement
The problem statement chosen for this project is **to predict fraudulent credit card transactions with the help of machine learning models.**

 

In this project, we will analyse customer-level data that has been collected and analysed during a research collaboration of Worldline and the Machine Learning Group. 

 

The data set is taken from the [Kaggle website](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and has a total of 2,84,807 transactions; out of these, 492 are fraudulent. Since the data set is highly imbalanced, it needs to be handled before model building.

 

## Business problem overview
For many banks, retaining high profitable customers is the number one business goal. Banking fraud, however, poses a significant threat to this goal for different banks. In terms of substantial financial losses, trust and credibility, this is a concerning issue to both banks and customers alike.


It has been estimated by [Nilson Report](https://nilsonreport.com/upload/content_promo/The_Nilson_Report_Issue_1164.pdf) that by 2020, banking frauds would account for $30 billion worldwide. With the rise in digital payment channels, the number of fraudulent transactions is also increasing in new and different ways. 

 

In the banking industry, credit card fraud detection using machine learning is not only a trend but a necessity for them to put proactive monitoring and fraud prevention mechanisms in place. Machine learning is helping these institutions to reduce time-consuming manual reviews, costly chargebacks and fees as well as denials of legitimate transactions.

 

## Understanding and defining fraud
Credit card fraud is any dishonest act or behaviour to obtain information without proper authorisation from the account holder for financial gain. Among different ways of committing frauds, skimming is the most common one, which is a way of duplicating information that is located on the magnetic strip of the card. Apart from this, the other ways are as follows:

+ Manipulation/alteration of genuine cards
+ Creation of counterfeit cards
+ Stealing/loss of credit cards
+ Fraudulent telemarketing
 

Data dictionary
The data set can be downloaded using this [link](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

 

The data set includes credit card transactions made by European cardholders over a period of two days in September 2013. **Out of a total of 2,84,807 transactions, 492 were fraudulent.** This data set is highly unbalanced, **with the positive class (frauds) accounting for 0.172% of the total transactions.** The data set has also been modified with principal component analysis (PCA) to maintain confidentiality. Apart from ‘time’ and ‘amount’, all the other features **(V1, V2, V3, up to V28)** are the principal components obtained using PCA. The feature 'time' contains the seconds elapsed between the first transaction in the data set and the subsequent transactions. The feature 'amount' is the transaction amount. The **feature 'class' represents class labelling,** and it takes the value of 1 in cases of fraud and 0 in others.

 

## Project pipeline
  The project pipeline can be briefly summarised in the following four steps:

  + **Data Understanding:** To prepare for building a final model, it's important to understand the data by using functions provided by pandas such as [.info()](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.info.html), [.shape()](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.shape.html) and [.describe()](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.describe.html). These functions can help identify the features present in the data and provide useful insights for feature selection. This process allows for informed decision-making when choosing which features to include in the final model.
  
  + **Exploratory data analytics (EDA):** Perform univariate and bivariate analyses of the data, followed by feature transformations, if necessary. For the current data set, because Gaussian variables are used, thers is no need to perform Z-scaling. However, we can check whether there is any skewness in the data and try to mitigate it, as it might cause problems during the model building phase.
  + **Train/Test split:** used train/test split  to check the performance of the models with unseen data. Here, for validation, used the stratified k-fold cross-validation method.
  
  + **Model building / hyperparameter tuning:** Build different models such as [LogisticRegressionCV()](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html), [SVM()](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html), [XGBClassifier()](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html), [RandomForestClassifier()](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html), And fine-tune their hyperparameters to get the desired level of performance on the given data set, various sampling techniques such as [SMOTE()](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html), [RandomOverSampler()](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.RandomOverSampler.html), [ADASYN()](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.ADASYN.html).
Model evaluation: Evaluated the models using appropriate evaluation metrics,Since the data is heavily imbalanced, it is is more important to identify the fraudulent transactions accurately than the non-fraudulent ones. So, sensitivity() and ROC_AUC() as appropriate metric for this problem.
