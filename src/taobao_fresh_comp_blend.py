#! python taobao_fresh_comp_blend.py start_from= user_cnt= slide= topk= min_proba=0.5 start=2014-11-12 end=2014-12-10 output=
# from common import *
# import os
# import numpy as np
# from sklearn import preprocessing
# from sklearn.utils import shuffle
# from sklearn.cross_validation import train_test_split
# from sklearn.preprocessing import OneHotEncoder
# from feature_selection import *
# from sklearn.cross_validation import StratifiedKFold  


# import GBDT
# from taking_sample import *
# import verify
# from feature_selection import *

# import sys
# from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier  
# from sklearn.linear_model import LogisticRegression  
 

# # CRITICAL 50
# # ERROR    40
# # WARNING  30
# # INFO     20
# # DEBUG    10
# # NOTSET   0


# #loadDataAndSaveToRedis()
# #loadTrainCategoryItemAndSaveToRedis()

# def trainModelWithSlideWindow(window_start_date, final_end_date, slide_windows_days, start_from, user_cnt, 
#                               n_estimators, nag_per_pos, max_depth, learning_rate, min_samples_split):

#     window_end_date = window_start_date + datetime.timedelta(slide_windows_days)

#     slide_feature_mat_dict = dict()
#     feature_importances = None

#     #滑动窗口 11-18 -- 12-17 得到特征的重要性， 并保留每个滑动窗口训练是使用的特征矩阵
#     while (window_end_date < final_end_date):    
#         params = {
#             'window_start_date' : window_start_date, 
#             'window_end_date' : window_end_date, 
#             'nag_per_pos' : nag_per_pos, 
#             'n_estimators' : n_estimators, 
#             'max_depth' : max_depth, 
#             'learning_rate' : learning_rate,
#             'min_samples_split' : min_samples_split,
#             'start_from' : start_from, 
#             'user_cnt' : user_cnt 
#         }
#         Xmat, Ymat, slide_window_clf = GBDT.GBDT_slideWindows(**params)

#         if (slide_window_clf == None):
#             window_end_date += datetime.timedelta(1)
#             window_start_date += datetime.timedelta(1)
#             continue

#         slide_feature_mat_dict[(window_start_date, window_end_date)] = (Xmat, Ymat, slide_window_clf)
#         if (feature_importances is None):
#             feature_importances = slide_window_clf.feature_importances_
#         else:
#             feature_importances += slide_window_clf.feature_importances_

#         window_end_date += datetime.timedelta(1)
#         window_start_date += datetime.timedelta(1)

#     return slide_feature_mat_dict, feature_importances


# ################################################################################################################
# ################################################################################################################
# ################################################################################################################
# ################################################################################################################


# print("--------------------------------------------------")
# print("--------------- Starting... ----------------------")
# print("--------------------------------------------------")


# loadTrainCategoryItemFromRedis()
# loadTestSet()


# start_from = int(sys.argv[1].split("=")[1])
# user_cnt = int(sys.argv[2].split("=")[1])
# slide_windows_days = int(sys.argv[3].split("=")[1])
# topK = int(sys.argv[4].split("=")[1])
# min_proba = float(sys.argv[5].split("=")[1])
# start_date = sys.argv[6].split("=")[1]
# end_date = sys.argv[7].split("=")[1]
# does_output = int(sys.argv[8].split("=")[1])
# print("start_from %d, user_cnt %d, slide window %d, topK %d, min prob %.2f, (%s, %s), output %d " % (start_from,
#       user_cnt, slide_windows_days, topK, min_proba, start_date, end_date, does_output))


# n_estimators = 50
# max_depth = 7
# learning_rate = 0.01
# min_samples_split = 2
# nag_per_pos = 10

# log_file = '..\\log\\log.%d.%d.txt' % (start_from, user_cnt)

# logging.basicConfig(level=logging.INFO,
#                     format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
#                     datefmt='%a, %d %b %Y %H:%M:%S',
#                     filename=log_file,
#                     filemode='w')

# print("=====================================================================")
# print("=== creating feature maxtrix...  user from %d, count %d, slide window size %d (%s -- %s)" % 
#       (start_from, user_cnt, slide_windows_days, start_date, end_date))
# print("=====================================================================")
# loadRecordsFromRedis(start_from, user_cnt)


# window_start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d").date()
# final_end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d").date()

# params = {
#     'window_start_date' : window_start_date, 
#     'final_end_date' : final_end_date, 
#     'slide_windows_days' : slide_windows_days, 
#     'start_from' : start_from, 
#     'user_cnt' : user_cnt,
#     'nag_per_pos' : nag_per_pos,    
#     'n_estimators' : n_estimators, 
#     'max_depth' : max_depth, 
#     'learning_rate' : learning_rate,
#     'min_samples_split' : min_samples_split,
# }

# slide_feature_mat_dict, feature_importance = trainModelWithSlideWindow(**params)


# window_end_date = final_end_date
# window_start_date = window_end_date - datetime.timedelta(slide_windows_days)
# print("=====================================================================")
# print("=== creating weight maxtrix... (%s -- %s)" % (window_start_date, window_end_date))
# print("=====================================================================")

# samples_weight, Ymat_weight = takeSamples(window_start_date, window_end_date, nag_per_pos, start_from, user_cnt)
# params = {'window_start_date' : window_start_date, 
#           'window_end_date' : window_end_date,
#           'nag_per_pos' : nag_per_pos, 
#           'samples' : samples_weight, 
#          }
# Xmat_weight = createFeatureMatrix(**params)
# # Xmat_weight_blend = np.zeros((Xmat_weight.shape[0], len(slide_feature_mat_dict)))
# Xmat_weight_blend = Xmat_weight

# forecast_date = final_end_date + datetime.timedelta(1)
# window_start_date = forecast_date - datetime.timedelta(slide_windows_days)
# print("=====================================================================")
# print("=== creating forecast maxtrix... (%s -- %s)" % (window_start_date, forecast_date))
# print("=====================================================================")
# params = {'window_start_date' : window_start_date,
#           'window_end_date' : forecast_date,
#           'user_records' : g_user_behavior_patten,
#           'during_forecasting' : True
#          }
# samples_forecast = takeSamplesByUserBehavior(**params)
# samples_forecast = list(samples_forecast)

# params = {'window_start_date' : window_start_date, 
#           'window_end_date' : forecast_date,
#           'nag_per_pos' : nag_per_pos, 
#           'samples' : samples_forecast, 
#          }
# Xmat_forecast = createFeatureMatrix(**params)
# # Xmat_forecast_blend = np.zeros((Xmat_forecast.shape[0], len(slide_feature_mat_dict)))
# Xmat_forecast_blend = Xmat_forecast
# n_folds = 5

# for j, (window_start_end_date, X_Y_mat_clf) in enumerate(slide_feature_mat_dict.items()):
#     window_start_date = window_start_end_date[0]
#     window_end_date = window_start_end_date[1]
#     X_mat = X_Y_mat_clf[0]
#     Y_mat = X_Y_mat_clf[1]

#     Y = np.zeros((len(Y_mat), 1))
#     Y[:, 0] = Y_mat

#     skf = list(StratifiedKFold(Y_mat, n_folds))

#     for i, (train, test) in enumerate(skf):
#         print ("%s Fold %s (%s, %s)\r" % (getCurrentTime(),i, window_start_date, window_end_date), end="")
#         Xmat_train = X_mat[train]
#         Xmat_test = X_mat[test]
#         Ymat_train = Y[train, 0]
#         Ymat_test = Y[test, 0]

#         Xmat_weight_blend_i = np.zeros((Xmat_weight.shape[0], n_folds))
#         Xmat_forecast_blend_i = np.zeros((Xmat_forecast.shape[0], n_folds))

#         params = {'n_estimators': n_estimators, 
#                   'max_depth': max_depth,
#                   'min_samples_split': min_samples_split,
#                   'learning_rate': learning_rate, 
#                   'loss': 'deviance'
#                   }
#         clf = GradientBoostingClassifier(**params)
#         clf.fit(Xmat_train, Ymat_train)

#         X_train_gbdt_leaf = clf.apply(Xmat_weight)[:, :, 0]
#         X_forecast_gbdt_leaf = clf.apply(Xmat_forecast)[:, :, 0]

#         onehot_val = [0 for x in range(n_estimators)]
#         max_apply = 0

#         for n in range(n_estimators):
#             onehot_val[n] = max(onehot_val[n], X_train_gbdt_leaf[:, n].max(), X_forecast_gbdt_leaf[:, n].max())
#             max_apply = max(max_apply, X_train_gbdt_leaf[:, n].max(), X_forecast_gbdt_leaf[:, n].max())

#         onehot_mat = np.zeros((max_apply + 1, n_estimators))
#         for n in range(n_estimators):
#             for o in range(int(onehot_val[n]) + 1):
#                 onehot_mat[o, n] = o

#         onehot_enc = OneHotEncoder()
#         onehot_enc.fit(onehot_mat)

#         X_train_gbdt_leaf = onehot_enc.transform(X_train_gbdt_leaf).toarray()
#         Xmat_weight_blend = np.column_stack((Xmat_weight_blend, X_train_gbdt_leaf))

#         X_forecast_gbdt_leaf = onehot_enc.transform(X_forecast_gbdt_leaf).toarray()
#         Xmat_forecast_blend = np.column_stack((Xmat_forecast_blend, X_forecast_gbdt_leaf))

#     #     # 用第 i 折的数据训练出的model来预测, 因为有 n_folds 折，所以会预测 n_folds 次,
#     #     # 做出的预测按行取平均值， 作为第 j 个model做出的预测
#     #     Xmat_weight_blend_i[:, i] = clf.predict_proba(Xmat_weight)[:, 1]
#     #     Xmat_forecast_blend_i[:, i] = clf.predict_proba(Xmat_forecast)[:, 1]

#     # Xmat_weight_blend[:,j] = Xmat_weight_blend_i.mean(1) # 相当于第 j 个model 对 Xmat_weight 做的转换
#     # Xmat_forecast_blend[:,j] = Xmat_forecast_blend_i.mean(1) # 相当于第 j 个model 对 Xmat_forecast 做的转换

# print("%s Xmat_weight_blend (%s, %s), Xmat_forecast_blend (%d, %d)" % (getCurrentTime(), 
#     Xmat_weight_blend.shape[0], Xmat_weight_blend.shape[1],
#     Xmat_forecast_blend.shape[0], Xmat_forecast_blend.shape[1]))

# logisticReg = LogisticRegression()
# logisticReg.fit(Xmat_weight_blend, Ymat_weight) #用转换之后的Xmat_weight来训练LR

# y_submission = logisticReg.predict_proba(Xmat_forecast_blend)[:,1]  

# print("Linear stretch of predictions to [0,1]")
# y_submission = (y_submission - y_submission.min()) / (y_submission.max() - y_submission.min())

# # 按照 probability 降序排序
# prob_desc = np.argsort(-y_submission)
# forecasted_user_item = []

# for i, prob_index in enumerate(prob_desc):
#     prob = y_submission[prob_index]
#     if (prob < min_proba):
#         print("        %s findal_predicted_prob[%d] %.4f < min_proba %.4f, breaking..." % (getCurrentTime(), i, prob, min_proba))
#         break
#     forecasted_user_item.append((samples_forecast[prob_index], prob))

# if (does_output == 1):
#     file_idx = 0
#     output_file_name = "%s\\..\\output\\subdata\\forecast.GBDT.LR.%d.%d.%d.%s.%d.csv" % (runningPath, slide_windows_days, start_from, user_cnt, datetime.date.today(), file_idx)
#     while (os.path.exists(output_file_name)):
#         file_idx += 1
#         output_file_name = "%s\\..\\output\\subdata\\forecast.GBDT.LR.%d.%d.%d.%s.%d.csv" % (runningPath, slide_windows_days, start_from, user_cnt, datetime.date.today(), file_idx)

#     print("        %s outputting %s" % (getCurrentTime(), output_file_name))
    
#     outputFile = open(output_file_name, encoding="utf-8", mode='w')
#     for user_item_prob in forecasted_user_item:
#         outputFile.write("%s,%s,%.4f\n" % (user_item_prob[0][0], user_item_prob[0][1], user_item_prob[1]))

#     outputFile.close()
# else:
#     acutal_buy = takingPositiveSamplesOnDate(forecast_date, True)
#     verify.calcuatingF1(forecast_date, forecasted_user_item, acutal_buy)

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
samples_train, Ymat = takeSamples(window_start_date, window_end_date, nag_per_pos, start_from, user_cnt)
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
print("%s samples count from buy %d / %d pattern" % (getCurrentTime(), len(samples_buy), len(samples_pattern)))

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