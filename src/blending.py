from __future__ import division  
import numpy as np  
import load_data  
from sklearn.cross_validation import StratifiedKFold  
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier  
from sklearn.linear_model import LogisticRegression  
from utility import *  
from evaluator import *  



def logloss(attempt, actual, epsilon=1.0e-15):  
"""Logloss, i.e. the score of the bioresponse competition. 
"""  
attempt = np.clip(attempt, epsilon, 1.0-epsilon)  
return - np.mean(actual * np.log(attempt) + (1.0 - actual) * np.log(1.0 - attempt))  


if __name__ == '__main__':  

np.random.seed(0) # seed to shuffle the train set  

# n_folds = 10  
n_folds = 5  
verbose = True  
shuffle = False  


# X, y, X_submission = load_data.load()  

train_x_id, train_x, train_y = preprocess_train_input()  
val_x_id, val_x, val_y = preprocess_val_input()  

X = train_x  
y = train_y  
X_submission = val_x  
X_submission_y = val_y  

if shuffle:  
    idx = np.random.permutation(y.size)  
    X = X[idx]  
    y = y[idx]  


skf = list(StratifiedKFold(y, n_folds))  

clfs = [RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='gini'),  
        RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='entropy'),  
        ExtraTreesClassifier(n_estimators=100, n_jobs=-1, criterion='gini'),  
        ExtraTreesClassifier(n_estimators=100, n_jobs=-1, criterion='entropy'),  
        GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=50)]  

print "Creating train and test sets for blending."  
  
dataset_blend_train = np.zeros((X.shape[0], len(clfs)))  
dataset_blend_test = np.zeros((X_submission.shape[0], len(clfs)))  
  
for j, clf in enumerate(clfs):  
    print j, clf  
    dataset_blend_test_j = np.zeros((X_submission.shape[0], len(skf)))  
    for i, (train, test) in enumerate(skf):  
        print "Fold", i  
        X_train = X[train]  
        y_train = y[train]  
        X_test = X[test]  
        y_test = y[test]  
        clf.fit(X_train, y_train)  # 用第 i 折的数据训练model j
        y_submission = clf.predict_proba(X_test)[:,1]  
        dataset_blend_train[test, j] = y_submission  # model j 对 test 预测出的结果
        dataset_blend_test_j[:, i] = clf.predict_proba(X_submission)[:,1]  # 用第 i 折的数据训练出的model来预测 
    dataset_blend_test[:,j] = dataset_blend_test_j.mean(1)  # 每一折都会训练出一个model，这些model在每一折上做出的预测按行取平均值， 作为第 j 个算法做出的预测
    print("val auc Score: %0.5f" % (evaluate2(dataset_blend_test[:,j], X_submission_y)))  # 评估在第 j 个算法做出的预测值与真实值

print  
print "Blending."  
# clf = LogisticRegression()  
clf = GradientBoostingClassifier(learning_rate=0.02, subsample=0.5, max_depth=6, n_estimators=100)  
clf.fit(dataset_blend_train, y)  
y_submission = clf.predict_proba(dataset_blend_test)[:,1]  

print "Linear stretch of predictions to [0,1]"  
y_submission = (y_submission - y_submission.min()) / (y_submission.max() - y_submission.min())  
print "blend result"  
print("val auc Score: %0.5f" % (evaluate2(y_submission, X_submission_y)))  
print "Saving Results."  
np.savetxt(fname='blend_result.csv', X=y_submission, fmt='%0.9f')  