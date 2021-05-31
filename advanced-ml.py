import sys
sys.path.insert(0, 'data')
from data.data_description import feature_names
import pandas as pd
import numpy as np
from matplotlib import pyplot
import collections
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedStratifiedKFold
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, f1_score, auc, accuracy_score, precision_score, recall_score
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, RandomOverSampler
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import TomekLinks
from imblearn.ensemble import BalancedBaggingClassifier
from imblearn.pipeline import Pipeline
from clover.over_sampling import ClusterOverSampler


global y_predicted
global lr_probs

#CURVES
fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8), (ax9, ax10)) = pyplot.subplots(5,2)
fig.suptitle('ROC AND AUC CURVES')   
fig.tight_layout(pad=1.0)

 
def makeClassificationRandomForest(X_train, y_train, X_test, y_test):
 global y_predicted
 global lr_probs 
 #random forest classifier with class-imbalance
 model = RandomForestClassifier(n_estimators=25, random_state=12)
 model.fit(X_train,y_train)
 y_predicted = model.predict(X_test)
 #Not relevant metrics
 print('Accuracy Score : %f'%accuracy_score(y_test,y_predicted))
 print('Precision Score : %f'%precision_score(y_test,y_predicted,average='macro'))
 print('Recall Score : %f' %recall_score(y_test,y_predicted,average='macro'))
 print('F1 Score : %f'%f1_score(y_test,y_predicted,average='macro'))
 lr_probs = model.predict_proba(X_test)
 lr_probs = lr_probs[:, 1]
 
def printCurvesWithClassImbalance(lr_probs, y_test, y_predicted):
 # keep probabilities for the positive outcome only
 # calculate scores
 lr_auc = roc_auc_score(y_test, lr_probs)
 # summarize scores
 print('Random Forest: ROC AUC=%.3f' % (lr_auc))
 # calculate roc curves
 lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
 # plot the roc curve for the model
 ax1.plot(lr_fpr, lr_tpr, marker='.', label='Random Forest')
 # axis labels
 ax1.set_xlabel('False Positive Rate')
 ax1.set_ylabel('True Positive Rate')
 ax1.set_title('ROC CURVE with class imbalance')

 # predict class values
 lr_precision, lr_recall, _ = precision_recall_curve(y_test, lr_probs)
 lr_f1, lr_auc = f1_score(y_test, y_predicted), auc(lr_recall, lr_precision)
 # summarize scores
 print('Random Forest: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))
 # plot the precision-recall curves
 ax2.plot(lr_recall, lr_precision, marker='.', label='Random Forest')
 # axis labels
 ax2.set_xlabel('Recall')
 ax2.set_ylabel('Precision')
 ax2.set_title('AUC CURVE with class imbalance')
 
def printCurvesWithSMOTE(lr_probs, y_test, y_predicted):
 # keep probabilities for the positive outcome only
 # calculate scores
 lr_auc = roc_auc_score(y_test, lr_probs)
 # summarize scores
 print('Random Forest: ROC AUC=%.3f' % (lr_auc))
 # calculate roc curves
 lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
 # plot the roc curve for the model
 ax3.plot(lr_fpr, lr_tpr, marker='.', label='Random Forest')
 # axis labels
 ax3.set_xlabel('False Positive Rate')
 ax3.set_ylabel('True Positive Rate')
 ax3.set_title('ROC CURVE with SMOTE')

 # predict class values
 lr_precision, lr_recall, _ = precision_recall_curve(y_test, lr_probs)
 lr_f1, lr_auc = f1_score(y_test, y_predicted,average='macro'), auc(lr_recall, lr_precision)
 # summarize scores
 print('Random Forest: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))
 # plot the precision-recall curves
 ax4.plot(lr_recall, lr_precision, marker='.', label='Random Forest')
 # axis labels
 ax4.set_xlabel('Recall')
 ax4.set_ylabel('Precision')
 ax4.set_title('AUC CURVE with SMOTE')
 
def printCurvesWithBorderLineSMOTE(lr_probs, y_test, y_predicted):
 # keep probabilities for the positive outcome only
 # calculate scores
 lr_auc = roc_auc_score(y_test, lr_probs)
 # summarize scores
 print('Random Forest: ROC AUC=%.3f' % (lr_auc))
 # calculate roc curves
 lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
 # plot the roc curve for the model
 ax5.plot(lr_fpr, lr_tpr, marker='.', label='Random Forest')
 # axis labels
 ax5.set_xlabel('False Positive Rate')
 ax5.set_ylabel('True Positive Rate')
 ax5.set_title('ROC CURVE with BorderLine SMOTE')

 # predict class values
 lr_precision, lr_recall, _ = precision_recall_curve(y_test, lr_probs)
 lr_f1, lr_auc = f1_score(y_test, y_predicted,average='macro'), auc(lr_recall, lr_precision)
 # summarize scores
 print('Random Forest: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))
 # plot the precision-recall curves
 ax6.plot(lr_recall, lr_precision, marker='.', label='Random Forest')
 # axis labels
 ax6.set_xlabel('Recall')
 ax6.set_ylabel('Precision')
 ax6.set_title('AUC CURVE with BorderLine SMOTE')
 
def printCurvesWithRandomOverSampler(lr_probs, y_test, y_predicted):
 # keep probabilities for the positive outcome only
 # calculate scores
 lr_auc = roc_auc_score(y_test, lr_probs)
 # summarize scores
 print('Random Forest: ROC AUC=%.3f' % (lr_auc))
 # calculate roc curves
 lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
 # plot the roc curve for the model
 ax7.plot(lr_fpr, lr_tpr, marker='.', label='Random Forest')
 # axis labels
 ax7.set_xlabel('False Positive Rate')
 ax7.set_ylabel('True Positive Rate')
 ax7.set_title('ROC CURVE with RandomOverSamler')

 # predict class values
 lr_precision, lr_recall, _ = precision_recall_curve(y_test, lr_probs)
 lr_f1, lr_auc = f1_score(y_test, y_predicted,average='macro'), auc(lr_recall, lr_precision)
 # summarize scores
 print('Random Forest: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))
 # plot the precision-recall curves
 ax8.plot(lr_recall, lr_precision, marker='.', label='Random Forest')
 # axis labels
 ax8.set_xlabel('Recall')
 ax8.set_ylabel('Precision')
 ax8.set_title('AUC CURVE with RandomOverSamler')

def printCurvesWithClusterOverSampler(lr_probs, y_test, y_predicted):
 # keep probabilities for the positive outcome only
 # calculate scores
 lr_auc = roc_auc_score(y_test, lr_probs)
 # summarize scores
 print('Random Forest: ROC AUC=%.3f' % (lr_auc))
 # calculate roc curves
 lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
 # plot the roc curve for the model
 ax9.plot(lr_fpr, lr_tpr, marker='.', label='Random Forest')
 # axis labels
 ax9.set_xlabel('False Positive Rate')
 ax9.set_ylabel('True Positive Rate')
 ax9.set_title('ROC CURVE with ClusterOverSampler')

 # predict class values
 lr_precision, lr_recall, _ = precision_recall_curve(y_test, lr_probs)
 lr_f1, lr_auc = f1_score(y_test, y_predicted,average='macro'), auc(lr_recall, lr_precision)
 # summarize scores
 print('Random Forest: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))
 # plot the precision-recall curves
 ax10.plot(lr_recall, lr_precision, marker='.', label='Random Forest')
 # axis labels
 ax10.set_xlabel('Recall')
 ax10.set_ylabel('Precision')
 ax10.set_title('AUC CURVE with ClusterOverSampler')

def plotCurves():
 # show the legend
 pyplot.legend()
 # show the plot
 pyplot.show()
 
# feature_names = []
# print(feature_names)
data = pd.read_csv('./data/MI.data', sep=",", names=feature_names, index_col=False)
data.replace("?", np.nan, inplace = True)

features_data = data[data.columns[1:112]]
output_data   = data[data.columns[119]]
features_data = pd.DataFrame(features_data)
output_data   = pd.DataFrame(output_data)

#print list of columns and number of NaN values
missing_data = features_data.isnull()
print(features_data.isnull().sum())
#plot for these columns
features_data.isnull().sum().reset_index(name="names").plot.bar(x='index', y='names', rot=90)

'''
features_data['AGE'] = pd.to_numeric(features_data['AGE'])
features_data['S_AD_KBRIG'] = pd.to_numeric(features_data['S_AD_KBRIG'])
features_data['D_AD_KBRIG'] = pd.to_numeric(features_data['D_AD_KBRIG'])
features_data['S_AD_ORIT'] = pd.to_numeric(features_data['S_AD_ORIT'])
features_data['D_AD_ORIT'] = pd.to_numeric(features_data['D_AD_ORIT'])
features_data['K_BLOOD'] = pd.to_numeric(features_data['K_BLOOD'])
features_data['Na_BLOOD'] = pd.to_numeric(features_data['Na_BLOOD'])
features_data['ALT_BLOOD'] = pd.to_numeric(features_data['ALT_BLOOD'])
features_data['AST_BLOOD'] = pd.to_numeric(features_data['AST_BLOOD'])
features_data['KFK_BLOOD'] = pd.to_numeric(features_data['KFK_BLOOD'])
features_data['L_BLOOD'] = pd.to_numeric(features_data['L_BLOOD'])
features_data['ROE'] = pd.to_numeric(features_data['ROE'])

#Alternative way to fill NaN values at columns
#features_data['AGE'] = features_data['AGE'].replace(np.nan, features_data['AGE'].np.mean())
#features_data['AGE'].interpolate(method='linear', direction = 'forward', inplace=True) 


# list of values of 'ColumnName' column
marks_list = features_data['AGE'].tolist()
#features_data['D_AD_KBRIG'] = features_data['D_AD_KBRIG'].apply(str)
#marks_list = features_data['D_AD_KBRIG'].tolist()
# show the list
print(marks_list)

#Write columns with NaN values in an Output.csv file to get names. I keep that in case we need it

null_colname=features_data.columns[features_data.isnull().any()].tolist() #find columns which returns True for null testing and convert the column names to list
null_colnum=len(null_colname)                       # take length of the above list

p=str(null_colnum)+"# of columns:"                  # initialize string in the format of required output
for i in range(0,null_colnum):                      #iterate over the list
  p=p+'Column-'+null_colname[i]+' '               # concatenate column names to the string


filepath="./"
text_file = open(filepath+"Output.csv", "w")        #export to csv
text_file.write("%s"% p)
text_file.close()

# list of values of 'ColumnName' column
marks_list = features_data['K_BLOOD'].tolist()

# show the list
print(marks_list)
'''

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

#drop columns with many NaN values - got it from plot
del X["IBS_NASL"]
del X["KFK_BLOOD"]

print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6, random_state=0)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

#------------------------


imputer = KNNImputer(weights='uniform',n_neighbors=5)

X_train = imputer.fit_transform(X_train)
X_test  = imputer.transform(X_test)

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
#------------------------


makeClassificationRandomForest(X_train, y_train, X_test, y_test)
printCurvesWithClassImbalance(lr_probs, y_test, y_predicted)

counter = collections.Counter(y_train)
print('Before SMOTE',counter)
smote = SMOTE(random_state=12)

X_train_sm,y_train_sm = smote.fit_resample(X_train, y_train)

counter = collections.Counter(y_train_sm)
print('After SMOTE',counter)

makeClassificationRandomForest(X_train_sm, y_train_sm, X_test, y_test)
printCurvesWithSMOTE(lr_probs, y_test, y_predicted)

counter = collections.Counter(y_train)
print('Before SMOTE Borderline',counter)

borderLineSMOTE = BorderlineSMOTE(kind='borderline-2', random_state=0)
X_train_sm_borderline,y_train_sm_borderline = borderLineSMOTE.fit_resample(X_train, y_train)

counter = collections.Counter(y_train_sm_borderline)
print('After SMOTE Borderline',counter)

makeClassificationRandomForest(X_train_sm_borderline, y_train_sm_borderline, X_test, y_test)
printCurvesWithBorderLineSMOTE(lr_probs, y_test, y_predicted)

counter = collections.Counter(y_train)
print('Before RandomOverSampler',counter)

oversample = RandomOverSampler(sampling_strategy='minority')
#oversample = RandomOverSampler(sampling_strategy=0.5)
X_over, y_over = oversample.fit_resample(X_train, y_train)

counter = collections.Counter(y_over)
print('After RandomOverSampler',counter)

makeClassificationRandomForest(X_over, y_over, X_test, y_test)
printCurvesWithRandomOverSampler(lr_probs, y_test, y_predicted)

counter = collections.Counter(y_train)
print('Before KMeans',counter)

smote = SMOTE(random_state= 12)
kmeans = KMeans(n_clusters=50, random_state=17)
kmeans_smote = ClusterOverSampler(oversampler=smote, clusterer=kmeans)

# Fit and resample imbalanced data
X_res, y_res = kmeans_smote.fit_resample(X_train, y_train)

counter = collections.Counter(y_res)
print('After KMeans',counter)

makeClassificationRandomForest(X_res, y_res, X_test, y_test)
printCurvesWithClusterOverSampler(lr_probs, y_test, y_predicted)

plotCurves()

'''
steps = [('over', RandomOverSampler()), ('model', RandomForestClassifier())]
pipeline = Pipeline(steps=steps)
# evaluate pipeline
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
scores = cross_val_score(pipeline, X_train, y_train, scoring='roc_auc', cv=cv, n_jobs=-1)

print('Mean ROC AUC: %.3f' % np.mean(scores))
'''
