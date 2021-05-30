import sys
sys.path.insert(0, 'data')
from data.data_description import feature_names
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import ensemble, metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
import numpy as np

# feature_names = []
# print(feature_names)
data = pd.read_csv('./data/MI.data', sep=",", names=feature_names, index_col=False)
data.replace("?", np.nan, inplace = True)

features_data = data[data.columns[1:112]]
output_data   = data[data.columns[119]]
features_data =pd.DataFrame(features_data)
output_data   = pd.DataFrame(output_data)

missing_data = features_data.isnull()
print(features_data.isnull().sum())

features_data.isnull().sum().reset_index(name="names").plot.bar(x='index', y='names', rot=90)


# print(data.shape)
print(data.describe())
print(data.head(10))
# print(data.columns)
data = pd.DataFrame(data)

features = feature_names[1:112]
output = feature_names[119]
print(features)
print(output)
X = data[features].copy()
y = data[output].copy()

print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6, random_state=0)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

#X_train.drop(columns=['R_AB_1_n', 'R_AB_2_n', 'R_AB_3_n', 'NA_R_1_n', 'NA_R_2_n', 'NA_R_3_n', 'NOT_NA_1_n', 'NOT_NA_2_n', 'NOT_NA_3_n'])
#X_test.drop(columns=['R_AB_1_n', 'R_AB_2_n', 'R_AB_3_n', 'NA_R_1_n', 'NA_R_2_n', 'NA_R_3_n', 'NOT_NA_1_n', 'NOT_NA_2_n', 'NOT_NA_3_n'])

#print(X_train.shape)
#print(X_test.shape)

#random forest classifier with class-imbalance

model = ensemble.RandomForestClassifier(criterion='entropy',n_estimators=3,max_depth=3)
model.fit(X_train,y_train)
y_predicted = model.predict(X_test)

lr_probs = model.predict_proba(X_test)

#Not relevant metrics
print('Accuracy Score : %f'%metrics.accuracy_score(y_test,y_predicted))
print('Precision Score : %f'%metrics.precision_score(y_test,y_predicted,average='macro'))
print('Recall Score : %f' %metrics.recall_score(y_test,y_predicted,average='macro'))
print('F1 Score : %f'%metrics.f1_score(y_test,y_predicted,average='macro'))

#CURVES


# keep probabilities for the positive outcome only
lr_probs = lr_probs[:, 1]
# calculate scores
lr_auc = roc_auc_score(y_test, lr_probs)
# summarize scores
print('Random Forest: ROC AUC=%.3f' % (lr_auc))
# calculate roc curves
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
# plot the roc curve for the model
pyplot.plot(lr_fpr, lr_tpr, marker='.', label='Random Forest')
# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
# show the legend
#pyplot.legend()
# show the plot
#pyplot.show()




# predict class values
yhat = model.predict(X_test)
lr_precision, lr_recall, _ = precision_recall_curve(y_test, lr_probs)
lr_f1, lr_auc = f1_score(y_test, yhat), auc(lr_recall, lr_precision)
# summarize scores
print('Logistic: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))
# plot the precision-recall curves
no_skill = len(y_test[y_test==1]) / len(y_test)
pyplot.plot(lr_recall, lr_precision, marker='.', label='Logistic')
# axis labels
pyplot.xlabel('Recall')
pyplot.ylabel('Precision')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()
