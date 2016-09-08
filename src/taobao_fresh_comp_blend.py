from common import *
import os
import numpy as np
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import OneHotEncoder
from feature_selection import *
from sklearn.cross_validation import StratifiedKFold  


import GBDT
from taking_sample import *
import verify
from feature_selection import *

import sys
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier  
from sklearn.linear_model import LogisticRegression  
 

# CRITICAL 50
# ERROR    40
# WARNING  30
# INFO     20
# DEBUG    10
# NOTSET   0


#loadDataAndSaveToRedis()
#loadTrainCategoryItemAndSaveToRedis()



print("--------------------------------------------------")
print("--------------- Starting... ----------------------")
print("--------------------------------------------------")


loadTrainCategoryItemFromRedis()
loadTestSet()


start_from = int(sys.argv[1].split("=")[1])
user_cnt = int(sys.argv[2].split("=")[1])
slide_windows_days = int(sys.argv[3].split("=")[1])
topK = int(sys.argv[4].split("=")[1])
min_proba = float(sys.argv[5].split("=")[1])
start_date = sys.argv[6].split("=")[1]
end_date = sys.argv[7].split("=")[1]
does_output = int(sys.argv[8].split("=")[1])
print("start_from %d, user_cnt %d, slide window %d, topK %d, min prob %.2f, (%s, %s), output %d " % (start_from,
      user_cnt, slide_windows_days, topK, min_proba, start_date, end_date, does_output))


n_estimators = 100
max_depth = 7
learning_rate = 0.01
min_samples_split = 2

nag_per_pos = 10

log_file = '..\\log\\log.%d.%d.txt' % (start_from, user_cnt)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename=log_file,
                    filemode='w')

print("=====================================================================")
print("  training...  user from %d, count %d, slide window size %d (%s -- %s)" % 
      (start_from, user_cnt, slide_windows_days, start_date, end_date))
print("=====================================================================")
loadRecordsFromRedis(start_from, user_cnt)


window_start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d").date()
window_end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d").date()
samples_train, Ymat = takeSamples(window_start_date, window_end_date, nag_per_pos, True, start_from, user_cnt)
params = {'window_start_date' : window_start_date, 
         'window_end_date' : window_end_date,
         'nag_per_pos' : nag_per_pos, 
         'samples' : samples_train, 
         }
n_folds = 10
skf = list(StratifiedKFold(Ymat, n_folds))  

Xmat_train = createFeatureMatrix(**params)
print("Xmat_train (%d, %d)" % (Xmat_train.shape[0], Xmat_train.shape[1]))

Ymat_train = np.zeros((len(Ymat), 1))
Ymat_train[:, 0] = Ymat
# print(Ymat_train)

forecast_date = window_end_date + datetime.timedelta(1)
window_start_date = window_start_date + datetime.timedelta(1)

print("%s taking samples for forecasting (%s, %s)" % (getCurrentTime(), window_start_date, forecast_date))

# samples_forecast, Ymat_forecast = takeSamples(window_start_date, forecast_date, nag_per_pos, True, start_from, user_cnt)

during_forecasting = False
if (forecast_date == datetime.datetime.strptime("2014-12-19", "%Y-%m-%d").date()):
    during_forecasting = True

params = {'window_start_date' : window_start_date,
          'window_end_date' : forecast_date,
          'user_records' : g_user_behavior_patten,
          'during_forecasting' : during_forecasting
}
samples_pattern = takeSamplesByUserBehavior(**params)


params = {'window_start_date' : window_start_date,
          'window_end_date' : forecast_date,
          'user_records' : g_user_buy_transection,
          'during_forecasting' : during_forecasting
}
samples_buy = takeSamplesByUserBehavior(**params)

samples_forecast = samples_pattern.union(samples_buy)
samples_forecast = list(samples_forecast)
print("%s samples forecast %d" % (getCurrentTime(), len(samples_forecast)))

params = {'window_start_date' : window_start_date, 
         'window_end_date' : forecast_date,
         'nag_per_pos' : nag_per_pos, 
         'samples' : samples_forecast, 
         }
Xmat_forecast = createFeatureMatrix(**params)
print("%s Xmat_forecast (%d, %d)" % (getCurrentTime(), Xmat_forecast.shape[0], Xmat_forecast.shape[1]))


clfs = [RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='gini'),  
        RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='entropy'),  
        ExtraTreesClassifier(n_estimators=100, n_jobs=-1, criterion='gini'),  
        ExtraTreesClassifier(n_estimators=100, n_jobs=-1, criterion='entropy'),  
        GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=50)]  


dataset_blend_train = np.zeros((Xmat_train.shape[0], len(clfs)))  
dataset_blend_forecast = np.zeros((Xmat_forecast.shape[0], len(clfs)))

for j, clf in enumerate(clfs):
    # print (getCurrentTime(), j, clf)
    dataset_blend_forecast_j = np.zeros((Xmat_forecast.shape[0], len(skf)))  
    for i, (train, test) in enumerate(skf):  
        print ("%s Fold %s \r" % (getCurrentTime(),i), end="")

        X_train = Xmat_train[train]  
        y_train = Ymat_train[train, 0]  
        X_test = Xmat_train[test]  
        y_test = Ymat_train[test, 0]  

        clf.fit(X_train, y_train)  # 用第 i 折的数据训练model j, 然后在test上进行预测
        dataset_blend_train[test, j] = clf.predict_proba(X_test)[:,1]  # model j 对 test 预测出的结果

        # 用第 i 折的数据训练出的model来预测, 因为有 n_folds 折，所以会预测 n_folds 次,
        # 做出的预测按行取平均值， 作为第 j 个model做出的预测
        dataset_blend_forecast_j[:, i] = clf.predict_proba(Xmat_forecast)[:,1]  
    dataset_blend_forecast[:,j] = dataset_blend_forecast_j.mean(1) # 相当于第 j 个model 对 Xmat_forecast 做的转换


clf = GradientBoostingClassifier(learning_rate=0.02, subsample=0.5, max_depth=6, n_estimators=100)  
clf.fit(dataset_blend_train, Ymat_train[:, 0])

y_submission = clf.predict_proba(dataset_blend_forecast)[:,1]  

print("Linear stretch of predictions to [0,1]")
y_submission = (y_submission - y_submission.min()) / (y_submission.max() - y_submission.min())

# 按照 probability 降序排序
prob_desc = np.argsort(-y_submission)
forecasted_user_item = []

for i, prob_index in enumerate(prob_desc):
    prob = y_submission[prob_index]
    if (prob < min_proba):
        print("        %s findal_predicted_prob[%d] %.4f < min_proba %.4f, breaking..." % (getCurrentTime(), i, prob, min_proba))
        break
    forecasted_user_item.append((samples_forecast[prob_index], prob))

if (does_output == 1):
    file_idx = 0
    output_file_name = "%s\\..\\output\\subdata\\forecast.GBDT.LR.%d.%d.%d.%s.%d.csv" % (runningPath, slide_windows_days, start_from, user_cnt, datetime.date.today(), file_idx)
    while (os.path.exists(output_file_name)):
        file_idx += 1
        output_file_name = "%s\\..\\output\\subdata\\forecast.GBDT.LR.%d.%d.%d.%s.%d.csv" % (runningPath, slide_windows_days, start_from, user_cnt, datetime.date.today(), file_idx)

    print("        %s outputting %s" % (getCurrentTime(), output_file_name))
    
    outputFile = open(output_file_name, encoding="utf-8", mode='w')
    for user_item_prob in forecasted_user_item:
        outputFile.write("%s,%s,%.4f\n" % (user_item_prob[0][0], user_item_prob[0][1], user_item_prob[1]))

    outputFile.close()
else:
    acutal_buy = takingPositiveSamplesOnDate(forecast_date, True)
    verify.calcuatingF1(forecast_date, forecasted_user_item, acutal_buy)