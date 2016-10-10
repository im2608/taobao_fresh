#! python taobao_fresh_comp_mean.py start_from= user_cnt= slide= topk= test=1 min_proba=0.5 start=2014-11-12 end=2014-12-10
from common import *
# import userCF
# import itemCF
# import apriori
import os
import numpy as np
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import (GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier)
from feature_selection import *
from sklearn.cross_validation import StratifiedKFold  




#滑动窗口 11-18 -- 12-17 得到并累加特征的重要性， 并保留每个滑动窗口训练时使用的特征矩阵
def trainModelWithSlideWindow(window_start_date, final_end_date, slide_windows_days, start_from, user_cnt, 
                              n_estimators, nag_per_pos, max_depth, learning_rate, min_samples_split):

    window_end_date = window_start_date + datetime.timedelta(slide_windows_days)

    slide_feature_mat_dict = dict()
    feature_importances = None

    #滑动窗口 11-18 -- 12-18 得到特征的重要性， 并保留每个滑动窗口训练是使用的特征矩阵
    while (window_end_date <= final_end_date):    
        params = {
            'window_start_date' : window_start_date, 
            'window_end_date' : window_end_date, 
            'nag_per_pos' : nag_per_pos, 
            'n_estimators' : n_estimators, 
            'max_depth' : max_depth, 
            'learning_rate' : learning_rate,
            'min_samples_split' : min_samples_split,
            'start_from' : start_from, 
            'user_cnt' : user_cnt 
        }
        Xmat, Ymat, slide_window_clf = GBDT.GBDT_slideWindows(**params)
        if (slide_window_clf == None):
            window_end_date += datetime.timedelta(1)
            window_start_date += datetime.timedelta(1)
            continue

        slide_feature_mat_dict[(window_start_date, window_end_date)] = (Xmat, Ymat, slide_window_clf)
        if (feature_importances is None):
            feature_importances = slide_window_clf.feature_importances_
        else:
            feature_importances += slide_window_clf.feature_importances_

        window_end_date += datetime.timedelta(1)
        window_start_date += datetime.timedelta(1)

    return slide_feature_mat_dict, feature_importances


################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################

import LR
import RF
import GBDT
from taking_sample import *
import verify
from feature_selection import *
from recommendation_same_day import *

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

window_start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d").date()
final_end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d").date()
forecast_date = final_end_date + datetime.timedelta(1)

loadRecordsFromRedis(start_from, user_cnt)

# countUserCategoryOnSameDay(window_start_date, final_end_date)

params = {
    'window_start_date' : window_start_date, 
    'final_end_date' : final_end_date, 
    'slide_windows_days' : slide_windows_days, 
    'start_from' : start_from, 
    'user_cnt' : user_cnt,
    'nag_per_pos' : nag_per_pos,
    'n_estimators' : n_estimators, 
    'max_depth' : max_depth, 
    'learning_rate' : learning_rate,
    'min_samples_split' : min_samples_split,
}
slide_feature_mat_dict, feature_importance = trainModelWithSlideWindow(**params)

logging.info("After slide window, feature_importance is : ")
useful_features_idx = []

for idx, importance in enumerate(feature_importance):
    if (feature_importance[idx] >= g_min_inportance):
        useful_features_idx.append(idx)
logging.info("useful_features_idx %s" % useful_features_idx)

print("=====================================================================")
print(" training with feature importance  (useful features: %d)  user from %d, count %d, slide window size %d " % 
      (len(useful_features_idx), start_from, user_cnt, slide_windows_days))
print("=====================================================================")

slide_windows_models = []
# 用满足 min importance 的特征重新训练模型, 训练好的模型保存在 slide_windows_models 中
for window_start_end_date, X_Y_mat_clf in slide_feature_mat_dict.items():
    window_start_date = window_start_end_date[0]
    window_end_date = window_start_end_date[1]
    X_mat = X_Y_mat_clf[0]
    Y_mat = X_Y_mat_clf[1]
    clf = X_Y_mat_clf[2]

    X_useful_mat = X_mat#[:, useful_features_idx]

    params = {
        'n_estimators': n_estimators, 
        'max_depth': max_depth,
        'min_samples_split': 2,
        'learning_rate': learning_rate, 
        #'loss': 'ls'
        'loss': 'deviance'
    }

    slide_windows_models.append((X_useful_mat, clf, window_start_end_date))

if (does_output == 0):
    verify_user_start = start_from + user_cnt
    verify_users = user_cnt * 2
    print("%s reloading verifying users... %d" % (getCurrentTime(), verify_users))
    loadRecordsFromRedis(verify_user_start, verify_users)

#取得用于预测的特征矩阵
window_start_date = forecast_date - datetime.timedelta(slide_windows_days)
print("%s taking samples for forecasting %s " % (getCurrentTime(), forecast_date))
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

params = {'window_start_date' : window_start_date, 
         'window_end_date' : forecast_date,
         'nag_per_pos' : nag_per_pos, 
         'samples' : samples_forecast, 
         }
Xmat_forecast = createFeatureMatrix(**params)

m, n = np.shape(Xmat_forecast)
print("        %s matrix for generating forecast matrix (%s, %s), (%d, %d)" % (getCurrentTime(), window_start_date, forecast_date, m, n))

findal_predicted_prob = np.zeros((Xmat_forecast.shape[0], 2))

print("Voting...")
ensemble_vote = np.zeros((Xmat_forecast.shape[0], len(slide_windows_models)))
# 滑动窗口训练出的model分别对特征矩阵进行预测，
for i, X_useful_mat_clf_model in enumerate(slide_windows_models):
    X_useful_mat = X_useful_mat_clf_model[0]
    clf_model = X_useful_mat_clf_model[1]
    slide_windows_start = X_useful_mat_clf_model[2][0]
    slide_windows_end = X_useful_mat_clf_model[2][1]

    predicted_prob = clf_model.predict(Xmat_forecast)
    ensemble_vote[:, i] = predicted_prob

prediction = []
for vote_i in range(ensemble_vote.shape[0]):
    ones = 0
    zeros = 0
    for model_j in range(ensemble_vote.shape[1]):
        if (ensemble_vote[vote_i, model_j] == 1):
            ones += 1
        elif (ensemble_vote[vote_i, model_j] == 0):
            zeros += 1
        else:
            print("WARNING, unknown class %d" % ensemble_vote[vote_i, model_j])

    if (ones > zeros):
        prediction.append((samples_forecast[vote_i], vote_i))

if (does_output == 1):
    print("=====================================================================")
    print(" forecasting %s  user from %d, count %d, slide window size %d" % 
         (forecast_date, start_from, user_cnt, slide_windows_days))
    print("=====================================================================")

    # 按照 probability 降序排序
    prob_desc = np.argsort(-findal_predicted_prob[:, 1])

    file_idx = 0
    output_file_name = "%s\\..\\output\\subdata\\forecast.GBDT.LR.%d.%d.%d.%s.%d.csv" % (runningPath, slide_windows_days, start_from, user_cnt, datetime.date.today(), file_idx)
    while (os.path.exists(output_file_name)):
        file_idx += 1
        output_file_name = "%s\\..\\output\\subdata\\forecast.GBDT.LR.%d.%d.%d.%s.%d.csv" % (runningPath, slide_windows_days, start_from, user_cnt, datetime.date.today(), file_idx)

    print("        %s outputting %s" % (getCurrentTime(), output_file_name))
    outputFile = open(output_file_name, encoding="utf-8", mode='w')
    for i, prob_index in enumerate(prob_desc):
        prob = findal_predicted_prob[prob_index, 1]
        if (prob < min_proba):
            print("        %s findal_predicted_prob[%d] %.4f < min_proba %.4f, breaking..." % (getCurrentTime(), i, prob, min_proba))
            break

        outputFile.write("%s,%s,%.4f\n" %
            (samples_forecast[prob_index][0], samples_forecast[prob_index][1], findal_predicted_prob[prob_index, 1] ))

    outputFile.close()

else:
    print("=====================================================================")
    print("== verifyPredictionEnsembleModel (%s, %s) esitmators %d, max depth %d, learning rate %.2f, min split %d" %
          (window_start_date, forecast_date, n_estimators, max_depth, learning_rate, min_samples_split))
    print("=====================================================================")

    acutal_buy = takingPositiveSamplesOnDate(forecast_date, True)
    verify.calcuatingF1(forecast_date, prediction, acutal_buy)
