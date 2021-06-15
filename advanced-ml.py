import sys
sys.path.insert(0, 'data')
import pandas as pd
import numpy as np
from matplotlib import pyplot
import collections
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedStratifiedKFold
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, f1_score, auc, accuracy_score, precision_score, recall_score, balanced_accuracy_score, plot_confusion_matrix
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, RandomOverSampler
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import TomekLinks
from clover.over_sampling import ClusterOverSampler
from sklearn.svm import SVC
from numpy import mean,where
from sklearn.decomposition import PCA
from imblearn.under_sampling import NearMiss
from xgboost import XGBClassifier
from imblearn.under_sampling import ClusterCentroids
from numpy import isnan
from sklearn.linear_model import LogisticRegression

global y_predicted
global lr_probs
global model

#CURVES
fig, ((ax1, ax2, axBar), (ax3, ax4,axBar2), (ax5, ax6,axBar3), (ax7, ax8,axBar4), (ax9, ax10,axBar5), (ax11, ax12, axBar6),(ax13, ax14,axBar7)) = pyplot.subplots(7,3)
fig.suptitle('ROC AND AUC CURVES')   
fig.tight_layout(pad=0.5)

def plotTargetClassValues(X,y,numberOfPlot): 
 fig2, ((axx1,axx2),(axx3,axx4),(axx5,axx6),(axx7,axx8)) = pyplot.subplots(4,2)
 fig2.suptitle ('Number Of target values')
 if (numberOfPlot == 1):
   axx1.set_title('Imbalanced Data')
   for label, _ in counter.items():
     row_ix = where(y == label)[0]
     axx1.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))
 elif (numberOfPlot == 2):
   axx2.set_title('SMOTE')

   for label, _ in counter.items():
     row_ix = where(y == label)[0]
     axx2.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))     
 elif (numberOfPlot == 3):
   axx3.set_title('Borderline SMOTE')
   
   for label, _ in counter.items():
     row_ix = where(y == label)[0]
     axx3.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))     
 elif (numberOfPlot == 4):
   axx4.set_title('RandomOverSampler')

   for label, _ in counter.items():
     row_ix = where(y == label)[0]
     axx4.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))     
 elif (numberOfPlot == 5):
   axx5.set_title('ClusterOverSampler')
   for label, _ in counter.items():
     row_ix = where(y == label)[0]
     axx5.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))     
 elif (numberOfPlot == 6):
   axx6.set_title('UnderSampling')

   for label, _ in counter.items():
     row_ix = where(y == label)[0]
     axx6.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))     
 elif (numberOfPlot == 7):
   axx7.set_title('ClusterCentroids')

   for label, _ in counter.items():
     row_ix = where(y == label)[0]
     axx7.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))     
 else:
     fig2.show()
 
def makeClassificationRandomForest(X_train, y_train, X_test, y_test):
 global y_predicted
 global lr_probs 
 global model
 #model = RandomForestClassifier(n_estimators=10, random_state=12,class_weight='balanced_subsample',criterion='entropy')
 model = LogisticRegression(random_state=0)
 model.fit(X_train,y_train)
 y_predicted = model.predict(X_test)
 #Not relevant metrics
 print('-------------------')
 print('Accuracy Score : %f'%accuracy_score(y_test,y_predicted))
 print('Balanced Accuracy Score : %f'%balanced_accuracy_score(y_test, y_predicted))
 print('Precision Score : %f'%precision_score(y_test,y_predicted,average='macro'))
 print('Recall Score : %f' %recall_score(y_test,y_predicted,average='macro'))
 print('F1 Score : %f'%f1_score(y_test,y_predicted,average='macro'))
 print('-------------------')
 lr_probs = model.predict_proba(X_test)
 lr_probs = lr_probs[:, 0] 
 
def makeClassificationCostSensitive(X_train, y_train):
 #random forest classifier with class-imbalance
 model = SVC(gamma='scale', class_weight='balanced')
 # define evaluation procedure
 cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
 # evaluate model
 scores = cross_val_score(model, X_train, y_train, scoring='roc_auc', cv=cv, n_jobs=-1)
 # summarize performance
 print('-------------------')
 print('Mean ROC AUC: %.3f' % mean(scores))
 print('-------------------')
 
def printCurvesWithClassImbalance(lr_probs, y_test, y_predicted):
 lr_auc = roc_auc_score(y_test, lr_probs)
 # summarize scores
 print('-------------------')
 print('Random Forest: ROC AUC=%.3f' % (lr_auc))
 print('-------------------')
 # calculate roc curves
 lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
 # plot the roc curve for the model
 ax1.plot(lr_fpr, lr_tpr, marker='.', label='Random Forest')
 ax1.set_xlabel('False Positive Rate')
 ax1.set_ylabel('True Positive Rate')
 ax1.set_title('ROC CURVE with class imbalance')

 # predict class values
 lr_precision, lr_recall, _ = precision_recall_curve(y_test, lr_probs)
 lr_f1, lr_auc = f1_score(y_test, y_predicted), auc(lr_recall, lr_precision)
 # summarize scores
 print('-------------------')
 print('Random Forest: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))
 print('-------------------')
 # plot the precision-recall curves
 ax2.plot(lr_recall, lr_precision, marker='.', label='Random Forest')
 ax2.set_xlabel('Recall')
 ax2.set_ylabel('Precision')
 ax2.set_title('AUC CURVE with class imbalance')
 plot_confusion_matrix(model, X_test, y_test, ax=axBar)
 
def printCurvesWithSMOTE(lr_probs, y_test, y_predicted):
 # calculate scores
 lr_auc = roc_auc_score(y_test, lr_probs)
 print('-------------------')
 print('Random Forest: ROC AUC=%.3f' % (lr_auc))
 print('-------------------')
 # calculate roc curves
 lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
 # plot the roc curve for the model
 ax3.plot(lr_fpr, lr_tpr, marker='.', label='Random Forest')
 ax3.set_xlabel('False Positive Rate')
 ax3.set_ylabel('True Positive Rate')
 ax3.set_title('ROC CURVE with SMOTE')

 # predict class values
 lr_precision, lr_recall, _ = precision_recall_curve(y_test, lr_probs)
 lr_f1, lr_auc = f1_score(y_test, y_predicted,average='macro'), auc(lr_recall, lr_precision)
 print('-------------------')
 print('Random Forest: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))
 print('-------------------')
 ax4.plot(lr_recall, lr_precision, marker='.', label='Random Forest')
 ax4.set_xlabel('Recall')
 ax4.set_ylabel('Precision')
 ax4.set_title('AUC CURVE with SMOTE')
 plot_confusion_matrix(model, X_test, y_test, ax=axBar2)
 
def printCurvesWithBorderLineSMOTE(lr_probs, y_test, y_predicted):
 # calculate scores
 lr_auc = roc_auc_score(y_test, lr_probs)
 print('-------------------')
 print('Random Forest: ROC AUC=%.3f' % (lr_auc))
 print('-------------------')
 lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
 ax5.plot(lr_fpr, lr_tpr, marker='.', label='Random Forest')
 # axis labels
 ax5.set_xlabel('False Positive Rate')
 ax5.set_ylabel('True Positive Rate')
 ax5.set_title('ROC CURVE with BorderLine SMOTE')

 # predict class values
 lr_precision, lr_recall, _ = precision_recall_curve(y_test, lr_probs)
 lr_f1, lr_auc = f1_score(y_test, y_predicted,average='macro'), auc(lr_recall, lr_precision)
 print('-------------------')
 print('Random Forest: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))
 print('-------------------')
 ax6.plot(lr_recall, lr_precision, marker='.', label='Random Forest')
 ax6.set_xlabel('Recall')
 ax6.set_ylabel('Precision')
 ax6.set_title('AUC CURVE with BorderLine SMOTE')
 plot_confusion_matrix(model, X_test, y_test, ax=axBar3)
 
def printCurvesWithRandomOverSampler(lr_probs, y_test, y_predicted):
 lr_auc = roc_auc_score(y_test, lr_probs)
 print('-------------------')
 print('Random Forest: ROC AUC=%.3f' % (lr_auc))
 print('-------------------')
 lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
 ax7.plot(lr_fpr, lr_tpr, marker='.', label='Random Forest')
 ax7.set_xlabel('False Positive Rate')
 ax7.set_ylabel('True Positive Rate')
 ax7.set_title('ROC CURVE with RandomOverSamler')

 lr_precision, lr_recall, _ = precision_recall_curve(y_test, lr_probs)
 lr_f1, lr_auc = f1_score(y_test, y_predicted,average='macro'), auc(lr_recall, lr_precision)
 print('-------------------')
 print('Random Forest: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))
 print('-------------------')
 ax8.plot(lr_recall, lr_precision, marker='.', label='Random Forest')
 ax8.set_xlabel('Recall')
 ax8.set_ylabel('Precision')
 ax8.set_title('AUC CURVE with RandomOverSamler')
 plot_confusion_matrix(model, X_test, y_test, ax=axBar4)

def printCurvesWithClusterOverSampler(lr_probs, y_test, y_predicted):
 lr_auc = roc_auc_score(y_test, lr_probs)
 print('-------------------')
 print('Random Forest: ROC AUC=%.3f' % (lr_auc))
 print('-------------------')
 lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
 ax9.plot(lr_fpr, lr_tpr, marker='.', label='Random Forest')
 ax9.set_xlabel('False Positive Rate')
 ax9.set_ylabel('True Positive Rate')
 ax9.set_title('ROC CURVE with ClusterOverSampler')

 lr_precision, lr_recall, _ = precision_recall_curve(y_test, lr_probs)
 lr_f1, lr_auc = f1_score(y_test, y_predicted,average='macro'), auc(lr_recall, lr_precision)
 print('-------------------')
 print('Random Forest: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))
 print('-------------------')
 ax10.plot(lr_recall, lr_precision, marker='.', label='Random Forest')
 ax10.set_xlabel('Recall')
 ax10.set_ylabel('Precision')
 ax10.set_title('AUC CURVE with ClusterOverSampler')
 plot_confusion_matrix(model, X_test, y_test, ax=axBar5)

def printCurvesWithUnderSampling(lr_probs, y_test, y_predicted):
 lr_auc = roc_auc_score(y_test, lr_probs)
 print('-------------------')
 print('Random Forest: ROC AUC=%.3f' % (lr_auc))
 print('-------------------')
 lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
 ax11.plot(lr_fpr, lr_tpr, marker='.', label='Random Forest')
 ax11.set_xlabel('False Positive Rate')
 ax11.set_ylabel('True Positive Rate')
 ax11.set_title('ROC CURVE with UnderSampling')

 lr_precision, lr_recall, _ = precision_recall_curve(y_test, lr_probs)
 lr_f1, lr_auc = f1_score(y_test, y_predicted,average='macro'), auc(lr_recall, lr_precision)
 print('-------------------')
 print('Random Forest: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))
 print('-------------------')
 ax12.plot(lr_recall, lr_precision, marker='.', label='Random Forest')
 ax12.set_xlabel('Recall')
 ax12.set_ylabel('Precision')
 ax12.set_title('AUC CURVE with UnderSampling')
 plot_confusion_matrix(model, X_test, y_test, ax=axBar6)

def printCurvesWithClusterCentroids(lr_probs, y_test, y_predicted):
 lr_auc = roc_auc_score(y_test, lr_probs)
 print('-------------------')
 print('Random Forest: ROC AUC=%.3f' % (lr_auc))
 print('-------------------')
 lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
 ax13.plot(lr_fpr, lr_tpr, marker='.', label='Random Forest')
 ax13.set_xlabel('False Positive Rate')
 ax13.set_ylabel('True Positive Rate')
 ax13.set_title('ROC CURVE with ClusterCentroids')

 lr_precision, lr_recall, _ = precision_recall_curve(y_test, lr_probs)
 lr_f1, lr_auc = f1_score(y_test, y_predicted,average='macro'), auc(lr_recall, lr_precision)
 print('-------------------')
 print('Random Forest: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))
 print('-------------------')
 ax14.plot(lr_recall, lr_precision, marker='.', label='Random Forest')
 ax14.set_xlabel('Recall')
 ax14.set_ylabel('Precision')
 ax14.set_title('AUC CURVE with ClusterCentroids')
 plot_confusion_matrix(model, X_test, y_test, ax=axBar7)
 
def plotCurves():
 # show the plot
 fig.show()
 
#------------------------
data = pd.read_csv('./data/Myocardial infarction complications Database.csv')
print('-------------------')
print (data)
print('-------------------')
data.replace("?", np.nan, inplace = True)
#------------------------

#------------------------

#print list of columns and number of NaN values
missing_data = data.isnull()
print('-------------------')
print(data.isnull().sum())
print('-------------------')
#plot for these columns
data.isnull().sum().reset_index(name="names").plot.bar(x='index', y='names', rot=90)
#------------------------

#------------------------
print('-------------------')
print(data.describe())
print(data.head(10))
print('-------------------')
data = pd.DataFrame(data)

X = data.iloc[:, 1:112]
y = data.iloc[:, 114]


#drop columns with many NaN values - got it from plot
del X["IBS_NASL"]
del X["KFK_BLOOD"]
del X["S_AD_KBRIG"]
del X["D_AD_KBRIG"]
del X["R_AB_3_n"]

del X["R_AB_2_n"]
#R_AB_3_n
del X["NA_R_2_n"]
del X["NA_R_3_n"]
del X["NOT_NA_2_n"]
del X["NOT_NA_3_n"]

print('-------------------')
print(X.shape)
print(y.shape)
#------------------------

#------------------------

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6, random_state=0)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
print('-------------------')
#------------------------


#------------------------
#preproccessing
imputer = KNNImputer(weights='uniform',n_neighbors=5)

X_train = imputer.fit_transform(X_train)
X_test  = imputer.transform(X_test)

print('-------------------')
print('Missing Values Train: %d' % isnan(X_train).sum())
print('Missing Values Test: %d' % isnan(X_test).sum())
print('-------------------')

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

pca = PCA(n_components=3)# adjust yourself
pca.fit(X_train)
X_train = pca.transform(X_train)
X_test = pca.transform(X_test)

#kpca = KernelPCA(kernel='poly',n_components=700,degree=3,gamma=0.1,max_iter=10000)
#X_train = kpca.fit_transform(X_train)
#X_test = kpca.transform(X_test)
#------------------------

#------------------------
makeClassificationRandomForest(X_train, y_train, X_test, y_test)
printCurvesWithClassImbalance(lr_probs, y_test, y_predicted)
plotTargetClassValues(X_train,y_train,1)
#------------------------

#------------------------

counter = collections.Counter(y_train)
print('-------------------')
print('Before SMOTE',counter)
smote = SMOTE(random_state=12)

X_train_sm,y_train_sm = smote.fit_resample(X_train, y_train)

counter = collections.Counter(y_train_sm)
print('After SMOTE',counter)
print('-------------------')

makeClassificationRandomForest(X_train_sm, y_train_sm, X_test, y_test)
printCurvesWithSMOTE(lr_probs, y_test, y_predicted)
plotTargetClassValues(X_train_sm,y_train_sm,2)
#------------------------

#------------------------
counter = collections.Counter(y_train)
print('-------------------')
print('Before SMOTE Borderline',counter)

borderLineSMOTE = BorderlineSMOTE(kind='borderline-2', random_state=0)
X_train_sm_borderline,y_train_sm_borderline = borderLineSMOTE.fit_resample(X_train, y_train)

counter = collections.Counter(y_train_sm_borderline)
print('After SMOTE Borderline',counter)
print('-------------------')

makeClassificationRandomForest(X_train_sm_borderline, y_train_sm_borderline, X_test, y_test)
printCurvesWithBorderLineSMOTE(lr_probs, y_test, y_predicted)
plotTargetClassValues(X_train_sm_borderline,y_train_sm_borderline,3)
#------------------------

#------------------------
counter = collections.Counter(y_train)
print('-------------------')
print('Before RandomOverSampler',counter)

oversample = RandomOverSampler(sampling_strategy='minority')
#oversample = RandomOverSampler(sampling_strategy=0.5)
X_over, y_over = oversample.fit_resample(X_train, y_train)

counter = collections.Counter(y_over)
print('After RandomOverSampler',counter)
print('-------------------')

makeClassificationRandomForest(X_over, y_over, X_test, y_test)
printCurvesWithRandomOverSampler(lr_probs, y_test, y_predicted)
plotTargetClassValues(X_over,y_over,4)
#------------------------

#------------------------
counter = collections.Counter(y_train)
print('-------------------')
print('Before KMeans',counter)

smote = SMOTE(random_state= 12)
kmeans = KMeans(n_clusters=25, random_state=17)
kmeans_smote = ClusterOverSampler(oversampler=smote, clusterer=kmeans)

# Fit and resample imbalanced data
X_res, y_res = kmeans_smote.fit_resample(X_train, y_train)

counter = collections.Counter(y_res)
print('After KMeans',counter)
print('-------------------')

makeClassificationRandomForest(X_res, y_res, X_test, y_test)
printCurvesWithClusterOverSampler(lr_probs, y_test, y_predicted)
plotTargetClassValues(X_res,y_res,5)
#------------------------

#------------------------
makeClassificationCostSensitive(X_train, y_train)
#------------------------


#------------------------
counter = collections.Counter(y_train)
print('-------------------')
print('Before UnderSampling',counter)
undersample = NearMiss(version=2, n_neighbors=5)

X_under, y_under = undersample.fit_resample(X_train, y_train)

counter = collections.Counter(y_under)
print('After UnderSampling',counter)
print('-------------------')

makeClassificationRandomForest(X_under, y_under, X_test, y_test)
printCurvesWithUnderSampling(lr_probs, y_test, y_predicted)
plotTargetClassValues(X_under,y_under,6)
#------------------------

#------------------------
boostingmodel = XGBClassifier(scale_pos_weight=100)
# define evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(boostingmodel, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
# summarize performance
print('-------------------')
print('Mean ROC AUC: %.5f' % mean(scores))
print('-------------------')
#------------------------



#------------------------
counter = collections.Counter(y_train)
print('-------------------')
print('Before ClusterCentroids',counter)

trans = ClusterCentroids(random_state=0)
X_resampled, y_resampled = trans.fit_sample(X_train, y_train)

counter = collections.Counter(y_resampled)
print('After ClusterCentroids',counter)
print('-------------------')
makeClassificationRandomForest(X_resampled, y_resampled, X_test, y_test)
printCurvesWithClusterCentroids(lr_probs, y_test, y_predicted)
plotTargetClassValues(X_resampled,y_resampled,7)
#------------------------

#------------------------
counter = collections.Counter(y_train)
print('-------------------')
print('Before TomekLinks',counter)
tl = TomekLinks(sampling_strategy='majority')
X_res, y_res= tl.fit_sample(X_train, y_train)

counter = collections.Counter(y_res)
print('After TomekLinks',counter)
print('-------------------')
makeClassificationRandomForest(X_res, y_res, X_test, y_test)
#------------------------

#------------------------
plotTargetClassValues(X,y,8)
plotCurves()
#------------------------

