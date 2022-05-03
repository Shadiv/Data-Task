import pandas as pd
import numpy as np
from scipy import rand
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import RFE
import statsmodels.api as sm
from sklearn import metrics

data_1 = pd.read_csv('model_data.csv')
pd.set_option('display.float_format', lambda x: '%.2f' % x)

df_2 = pd.DataFrame()
df_2['customer_id'] = data_1['customerid']
df_2['score'] = data_1['creditscore']
df_2['age'] = data_1['age']
df_2['tenure'] = data_1['tenure']
df_2['nofproducts'] = data_1['numofproducts']
df_2['card'] = data_1['hascrcard']
df_2['exited'] = data_1['exited']

df_2['card'] = np.where(df_2['card'] == 't', 1, 0)
df_2['exited'] = np.where(df_2['exited'] == 't', 1, 0)

df_2 = df_2.astype({"score": int},
                   errors='raise')  # encountered error in score categorization (no logical operators between str and int), so I fixed score datatype

# For data analysis Logistic Regression model is chosen.
# We predict binary output (whether client will exit or not), does not require complex calculations (like decision trees), efficient to train and easy to interpret.                      

# classification of score columns into brackets
# Score: >750 (A), >620 (B), >490 (C), <490 (D)
df_2['score_cat'] = ""
df_2['score_cat'] = np.where(df_2['score'] < 490, 'D', df_2['score_cat'])
df_2['score_cat'] = np.where((df_2['score'] > 489) & (df_2['score'] < 620), 'C', df_2['score_cat'])
df_2['score_cat'] = np.where((df_2['score'] > 619) & (df_2['score'] < 750), 'B', df_2['score_cat'])
df_2['score_cat'] = np.where(df_2['score'] > 749, 'A', df_2['score_cat'])

# converting score categories into dummy columns

df_2['sc_a'] = np.where(df_2['score_cat'] == 'A', 1, 0)
df_2['sc_b'] = np.where(df_2['score_cat'] == 'B', 1, 0)
df_2['sc_c'] = np.where(df_2['score_cat'] == 'C', 1, 0)
df_2['sc_d'] = np.where(df_2['score_cat'] == 'D', 1, 0)

# age categorization into dummy columns (Groups: Below 25, below 45, below 65, above 65)

df_2['age_1'] = np.where(df_2['age'] <= 25, 1, 0)
df_2['age_2'] = np.where(df_2['age'] <= 45, 1, 0)
df_2['age_3'] = np.where(df_2['age'] <= 65, 1, 0)
df_2['age_4'] = np.where(df_2['age'] > 65, 1, 0)

# tenure categorization into dummy columns (Groups: new (<=2 year), regular(>2), loyal (>5) )

df_2['tenure_n'] = np.where(df_2['tenure'] < 2, 1, 0)
df_2['tenure_r'] = np.where((df_2['tenure'] > 2) & (df_2['tenure'] <= 5), 1, 0)
df_2['tenure_l'] = np.where(df_2['tenure'] > 5, 1, 0)

# numofproducts categorization into dummy columns (Two groups: 1, >1)

df_2['single_p'] = np.where(df_2['nofproducts'] == 1, 1, 0)
df_2['mult_p'] = np.where(df_2['nofproducts'] > 1, 1, 0)

# Creating DataFrame for LogReg model - only boolean features

cat_vars = ['customer_id', 'score', 'age', 'tenure', 'nofproducts', 'score_cat']
col_names = df_2.columns.values.tolist()

data_to_model = [i for i in col_names if i not in cat_vars]

df_3 = df_2[data_to_model]

# Check balance of data.

e_counts = df_3['exited'].value_counts()
print(e_counts)
# plt.bar(df_2['exited'].unique(), e_counts, align='center')
# plt.ylabel('customers')
# plt.title('Exited to not exited customers ratio')
# plt.show()

# Now we see that there is only 2040 clients who left bank in the past and 7956 clients who never did.
# For model we need more balanced data (at least that is what google says). To train model on balanced dataset we will use oversampling.


X = df_3.loc[:, df_3.columns != 'exited']  # Basic DF (x) without "exited"
y = df_3.loc[:, df_3.columns == 'exited']  # With exited column

oversample = SMOTE(random_state=0)  # Synthetic Minority OverSampling for balancing ratio exited / not exited
# We are splitting our dataframe into test / train parts. Then we take training part (x and y) and then create balanced oversample dataframes using SMOTE - Oversampling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
columns = X_train.columns
os_data_X, os_data_y = oversample.fit_resample(X_train, y_train)

os_data_X = pd.DataFrame(data=os_data_X, columns=columns)
os_data_y = pd.DataFrame(data=os_data_y, columns=['exited'])

# Check whether oversampled data are balanced now:

print("Total oversample records ", len(os_data_X))
print("Number of exited customers", len(os_data_y[os_data_y['exited'] == 1]))
print("Number of customers who did not exit", len(os_data_y[os_data_y['exited'] == 0]))
print("Ratio of exited customers in OS records ", len(os_data_y[os_data_y['exited'] == 1]) / len(os_data_X))
print("Ratio of customers who never exited in OS records", len(os_data_y[os_data_y['exited'] == 0]) / len(os_data_X))

# Reducing number of features to most important ones using RFE from sklearn

logreg_training = LogisticRegression(solver='lbfgs', max_iter=10000)
rfe = RFE(logreg_training)
rfe.max_iter = 10000  # Avoiding warning on breaching max iter limit.
rfe = rfe.fit(os_data_X, os_data_y.values.ravel())

print(rfe.support_)
print(rfe.ranking_)
print(os_data_X.columns)  # makes visible what columns should be eliminated

# Reducing  features to those chosen by recursive feature elemenation

rfe_cols = ['sc_a', 'sc_b', 'sc_c', 'sc_d', 'age_1', 'age_2', 'age_4']

X = os_data_X[rfe_cols]
y = os_data_y['exited']

# Significance check

stat_model = sm.Logit(y, X)
result = stat_model.fit()
# print(result.summary2())

# All p values are be low 5% - no changes needed.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,
                                                    random_state=0)  # again performing test_train split on sample sizes with most valuable features

# Final model

logreg_def = LogisticRegression()
logreg_def.fit(X_train, y_train)

y_predict = logreg_def.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg_def.score(X_test, y_test)))

# Print confusion matrix to show correct and incorrect predictions

confusion_matrix = metrics.confusion_matrix(y_test, y_predict)
print(confusion_matrix)

# Graphic output for exit prediction model. Building ROC curve. ROC curve summarizes the prediction performance of a classification model at all classification tresholds. Shows relationsip between False Positive Rate (X) and True Positive Rate (Y)

# TPR = Sensitivity (TP/(TP+FN))
# FPR = 1 - specifisity      (FP/(FP+TN))

rand_probability = [0 for _ in range(len(y_test))]
logreg_prob = logreg_def.predict_proba(X_test)

logreg_prob = logreg_prob[:,1]

random_auc = metrics.roc_auc_score(y_test, rand_probability) # area under curve score
logreg_auc = metrics.roc_auc_score(y_test, logreg_prob)

r_fpr, r_tpr, _ = metrics.roc_curve(y_test, rand_probability) 
l_fpr, l_tpr, _ = metrics.roc_curve(y_test, logreg_prob)

# plot

plt.plot(r_fpr, r_tpr, linestyle='--', label='Random prediction AUROC = %0.2f' % random_auc)
plt.plot(l_fpr, l_tpr, linestyle='dotted', label='Logistic Regression prediction AUROC = %0.2f' % logreg_auc) 
plt.title('ROC customer exit prediction model')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()
